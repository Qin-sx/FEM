#include <gmsh.h>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <fstream>

class Constitutive {
public:
    Constitutive(double E, double nv) : E(E), nv(nv) {
        D = cal_D();
        tangent = cal_tangent();
    }

    Eigen::Matrix3d cal_D() {
        Eigen::Matrix3d D;
        D << 1.0, nv, 0.0,
             nv, 1.0, 0.0,
             0.0, 0.0, 0.5 * (1.0 - nv);
        return (E / (1.0 - nv * nv)) * D;
    }

    Eigen::Matrix3d cal_tangent() {
        return D;
    }

    Eigen::Vector3d get_stress(const Eigen::Vector3d &strain) {
        return D * strain;
    }

	Eigen::Matrix3d get_D() const {
        return D;
    }

private:
    double E;
    double nv;
    Eigen::Matrix3d D;
    Eigen::Matrix3d tangent;
};

class Element {
public:
    Element(const Constitutive& material, const Eigen::MatrixXd& positions) 
        : D(material.get_D()) {
        x = positions.col(0);
        y = positions.col(1);
        node_set = Eigen::Vector3i(0, 1, 2);
        area_2 = cal_area_2();
        B = cal_B();
        K_element = element_integrate();
    }

    double cal_area_2() const {
        Eigen::Matrix3d matrix;
        matrix << 1.0, x(0), y(0),
                  1.0, x(1), y(1),
                  1.0, x(2), y(2);
        double det = matrix.determinant();
        return std::abs(det);
    }

    Eigen::MatrixXd cal_B() const {
        Eigen::MatrixXd B(3, 6);
        B << y(1) - y(2), 0.0, y(2) - y(0), 0.0, y(0) - y(1), 0.0,
             0.0, x(2) - x(1), 0.0, x(0) - x(2), 0.0, x(1) - x(0),
             x(2) - x(1), y(1) - y(2), x(0) - x(2), y(2) - y(0), x(1) - x(0), y(0) - y(1);
        return 0.5 / area_2 * B;
    }

    Eigen::MatrixXd element_integrate() const {
        return area_2 * B.transpose() * D * B;
    }

    void set_node_set(const Eigen::Vector3i& node_set) {
        this->node_set = node_set;
    }

    Eigen::Vector3d get_strain(const Eigen::VectorXd& deform_local) const {
        return B * deform_local;
    }

    Eigen::Vector3d get_stress(const Eigen::VectorXd& deform_local) const {
        return D * B * deform_local;
    }

// private:
public:
    Eigen::VectorXd x;
    Eigen::VectorXd y;
    Eigen::Vector3i node_set;
    Eigen::Matrix3d D;
    double area_2;
    Eigen::MatrixXd B;
    Eigen::MatrixXd K_element;
};

class GlobalStiffness {
public:
    GlobalStiffness(const std::vector<std::array<double, 3>>& points, 
                    const std::vector<std::array<std::size_t, 3>>& elements,
                    const Constitutive& materials) {
        // Get mesh data
        nodes.resize(points.size(), 2);
        for (std::size_t i = 0; i < points.size(); ++i) {
            nodes(i, 0) = points[i][0];
            nodes(i, 1) = points[i][1];
        }
        this->elements.resize(elements.size(), 3);
        for (std::size_t i = 0; i < elements.size(); ++i) {
            this->elements(i, 0) = elements[i][0];
            this->elements(i, 1) = elements[i][1];
            this->elements(i, 2) = elements[i][2];
        }
        // Initialize materials
		this->materials.reserve(elements.size());
		for (int element_index = 0; element_index < elements.size(); ++element_index) {
			this->materials.emplace_back(materials);
		}
        // Initialize global variables
        len_global = 2 * nodes.rows();
        deform = Eigen::VectorXd::Zero(len_global);
        force = Eigen::VectorXd::Zero(len_global);
        cal_K();
    }

    Element instant_Element(int element_index, const Eigen::Vector3i& node_set) {
        Constitutive material = materials[element_index];
        Eigen::MatrixXd positions(3, 2);
        for (int j = 0; j < 3; ++j) {
            positions.row(j) = nodes.row(node_set[j]);
        }
        return Element(material, positions);
    }

    void Ke2K(const Element& element) {
        Eigen::MatrixXd K_element = element.K_element;
        Eigen::Vector3i node_set = element.node_set;
        Eigen::VectorXi deform_global_index(6);
        for (int i = 0; i < 3; ++i) {
            deform_global_index(2 * i) = 2 * node_set(i);
            deform_global_index(2 * i + 1) = 2 * node_set(i) + 1;
        }
        for (int i_local = 0; i_local < 6; ++i_local) {
            int i_global = deform_global_index(i_local);
            for (int j_local = 0; j_local < 6; ++j_local) {
                int j_global = deform_global_index(j_local);
                K(i_global, j_global) += K_element(i_local, j_local);
            }
        }
    }

    void cal_K() {
        // Initialize K
        K = Eigen::MatrixXd::Zero(len_global, len_global);
        // Fill K per element
        for (int element_index = 0; element_index < elements.rows(); ++element_index) {
            Eigen::Vector3i node_set = elements.row(element_index);
            Element element = instant_Element(element_index, node_set);
            element.set_node_set(node_set);
            Ke2K(element);
        }
    }

public:
    Eigen::MatrixXd nodes;
    Eigen::MatrixXi elements;
    std::vector<Constitutive> materials;
    int len_global;
    Eigen::VectorXd deform;
    Eigen::VectorXd force;
    Eigen::MatrixXd K;
};

class ReducedStiffness : public GlobalStiffness {
public:
    ReducedStiffness(const std::vector<std::array<double, 3>>& points, 
                     const std::vector<std::array<std::size_t, 3>>& elements,
                     const Constitutive& materials)
        : GlobalStiffness(points, elements, materials) {
        // Initialize conditions
        x_fix = {};
        y_fix = {};
        f_given = {};
    }

    void mark_deform_free() {
        deform_free_index.clear();
        for (int node = 0; node < this->nodes.rows(); ++node) {
            if (x_fix.find(node) == x_fix.end()) {
                deform_free_index.push_back(2 * node);
            }
            if (y_fix.find(node) == y_fix.end()) {
                deform_free_index.push_back(2 * node + 1);
            }
        }
        len_reduce = deform_free_index.size();
    }

    void init_global_variables() {
        for (const auto& it : x_fix) {
			const auto& node = it.first;
			const auto& value = it.second;
            this->deform(2 * node) = value;
        }
        for (const auto& it : y_fix) {
			const auto& node = it.first;
			const auto& value = it.second;
            this->deform(2 * node + 1) = value;
        }
        for (const auto& it : f_given) {
			const auto& node = it.first;
			const auto& value = it.second;
            force(2 * node) = value[0];
            force(2 * node + 1) = value[1];
        }
    }

    void init_reduce_variables() {
        deform_reduce = Eigen::VectorXd::Zero(len_reduce);
        force_reduce = Eigen::VectorXd::Zero(len_reduce);
        K_reduce = Eigen::MatrixXd::Zero(len_reduce, len_reduce);
        for (int i_reduce = 0; i_reduce < len_reduce; ++i_reduce) {
            int i_global = deform_free_index[i_reduce];
            force_reduce(i_reduce) = this->force(i_global);
        }
        for (int i_reduce = 0; i_reduce < len_reduce; ++i_reduce) {
            int i_global = deform_free_index[i_reduce];
            for (int j_reduce = 0; j_reduce < len_reduce; ++j_reduce) {
                int j_global = deform_free_index[j_reduce];
                K_reduce(i_reduce, j_reduce) = this->K(i_global, j_global);
            }
        }
    }

    void reduce_system() {
        mark_deform_free();
        init_global_variables();
        init_reduce_variables();
    }

    void solve_reduce_system() {
        deform_reduce = K_reduce.colPivHouseholderQr().solve(force_reduce);
    }

    void update_global_variables() {
        for (int i_reduce = 0; i_reduce < len_reduce; ++i_reduce) {
            int i_global = deform_free_index[i_reduce];
            this->deform(i_global) = deform_reduce(i_reduce);
        }
    }

    void update_node_positions() {

        Eigen::MatrixXd original_positions = nodes;

        // Update nodes' positions
        Eigen::MatrixXd updated_positions = original_positions;
        for (int i = 0; i < nodes.rows(); ++i) {
            updated_positions(i, 0) += this->deform(2 * i);
            updated_positions(i, 1) += this->deform(2 * i + 1);
        }

        nodes = updated_positions;
    }

	void apply_conditions(
		std::unordered_map<int, double> x_fix,
		std::unordered_map<int, double> y_fix,
		std::unordered_map<int, std::array<double, 2>> f_given)
	{
		for (const auto& it : f_given) {
			const auto& index = it.first;
			const auto& force = it.second;
            this->f_given[index] = force;
        }
		for (const auto& it : x_fix) {
			const auto& index = it.first;
			const auto& value = it.second;
            this->x_fix[index] = value;
        }
        for (const auto& it : y_fix) {
			const auto& index = it.first;
			const auto& value = it.second;
            this->y_fix[index] = value;
        }
	}

	void export_to_gmsh(const std::string &filename, int mesh_size = 0.1)
	{
		gmsh::initialize();
		gmsh::model::add("ReducedStiffnessMesh");

		// Add nodes
		for (size_t i = 0; i < nodes.rows(); ++i)
		{
			gmsh::model::geo::addPoint(nodes(i,0), nodes(i,1), 0.0, mesh_size, i + 1);
		}

		// Add elements
		for (size_t i = 0; i < elements.rows(); ++i)
		{
			int p1 = elements(i, 0) + 1;
			int p2 = elements(i, 1) + 1;
			int p3 = elements(i, 2) + 1;

			int l1 = gmsh::model::geo::addLine(p1, p2);
			int l2 = gmsh::model::geo::addLine(p2, p3);
			int l3 = gmsh::model::geo::addLine(p3, p1);

			int cl = gmsh::model::geo::addCurveLoop({l1, l2, l3});
			gmsh::model::geo::addPlaneSurface({cl});
		}

		gmsh::model::geo::synchronize();
		gmsh::write(filename);
		gmsh::finalize();
	}

	void export_to_tecplot(const std::string &filename)
	{
		std::ofstream tecplotFile(filename);

		if (!tecplotFile.is_open())
		{
			std::cerr << "Error opening file: " << filename << std::endl;
			return;
		}

		// Write Tecplot header
		tecplotFile << "TITLE = \"Reduced Stiffness Mesh\"\n";
		tecplotFile << "VARIABLES = \"X\", \"Y\", \"DeformX\", \"DeformY\"\n";
		tecplotFile << "ZONE T=\"Zone 1\", N=" << nodes.rows() << ", E=" << elements.rows() << ", F=FEPOINT, ET=TRIANGLE\n";

		// Write node data
		for (size_t i = 0; i < nodes.rows(); ++i)
		{
			double deformX = deform(2 * i);
			double deformY = deform(2 * i + 1);
			tecplotFile << nodes(i, 0) << " " << nodes(i, 1) << " " << deformX << " " << deformY << "\n";
		}

		// Write element data
		for (size_t i = 0; i < elements.rows(); ++i)
		{
			tecplotFile << elements(i, 0) + 1 << " " << elements(i, 1) + 1 << " " << elements(i, 2) + 1 << "\n";
		}

		tecplotFile.close();
	}

public:
    std::unordered_map<int, double> x_fix;
    std::unordered_map<int, double> y_fix;
    std::unordered_map<int, std::array<double, 2>> f_given;
    std::vector<int> deform_free_index;
    int len_reduce;
    Eigen::VectorXd deform_reduce;
    Eigen::VectorXd force_reduce;
    Eigen::MatrixXd K_reduce;
};

int main(int argc, char **argv)
{
    {
        // Initialize Gmsh
        gmsh::initialize();

        // Create a new model
        gmsh::model::add("circle");

        // Define the center and radius of the circle
        std::array<double, 2> x0 = {0.0, 0.0};
        double radius = 1.0;
        double mesh_size = 0.1;
        int num_sections = 36;

        // Add the center point
        int centerTag = gmsh::model::geo::addPoint(x0[0], x0[1], 0.0, mesh_size);

        // Add points around the circle
        std::vector<int> pointTags;
        for (int i = 0; i < num_sections; ++i)
        {
            double angle = 2 * M_PI * i / num_sections;
            double x = x0[0] + radius * cos(angle);
            double y = x0[1] + radius * sin(angle);
            pointTags.push_back(gmsh::model::geo::addPoint(x, y, 0.0, mesh_size));
        }

        // Add lines to form the circle
        std::vector<int> lineTags;
        for (int i = 0; i < num_sections; ++i)
        {
            lineTags.push_back(gmsh::model::geo::addLine(pointTags[i], pointTags[(i + 1) % num_sections]));
        }
        // std::cout << "lineTags.size() = " << lineTags.size() << std::endl;
        // Create a curve loop and a plane surface
        int curveLoopTag = gmsh::model::geo::addCurveLoop(lineTags);
        gmsh::model::geo::addPlaneSurface({curveLoopTag});

        // Synchronize the model
        gmsh::model::geo::synchronize();

        // Generate the mesh
        gmsh::model::mesh::generate(2);

        // Write the mesh to a file
        gmsh::write("circle.msh");

        // // Finalize Gmsh
        // // gmsh::finalize();
    }

    // {
    //     // Initialize Gmsh
    //     gmsh::initialize();

    //     // Create a new model
    //     gmsh::model::add("polygon");

    //     // Define the points of the polygon
    //     std::vector<std::array<double, 3>> points = {
    //         {0.0, 0.0, 0.0},
    //         {1.0, 0.0, 0.0},
    //         {1.0, 1.0, 0.0},
    //         {0.0, 1.0, 0.0}};

    //     // Add points to the model
    //     std::vector<int> pointTags;
    //     for (const auto &point : points)
    //     {
    //         pointTags.push_back(gmsh::model::geo::addPoint(point[0], point[1], point[2], 1.0));
    //     }

    //     // Add lines to form the polygon
    //     std::vector<int> lineTags;
    //     for (size_t i = 0; i < points.size(); ++i)
    //     {
    //         std::cout << i << " " << pointTags[i] << std::endl;
    //         lineTags.push_back(gmsh::model::geo::addLine(pointTags[i], pointTags[(i + 1) % points.size()]));
    //         std::cout << "lineTags " << i << " " << lineTags[i] << std::endl;
    //     }

    //     // Create a curve loop and a plane surface
    //     int curveLoopTag = gmsh::model::geo::addCurveLoop(lineTags);
    //     gmsh::model::geo::addPlaneSurface({curveLoopTag});

    //     // Synchronize the model
    //     gmsh::model::geo::synchronize();

    //     // Generate the mesh
    //     gmsh::model::mesh::generate(2);

    //     // Write the mesh to a file
    //     gmsh::write("polygon.msh");

    //     Finalize Gmsh
    //     gmsh::finalize();
    // }

    // Get the mesh data
    std::vector<std::size_t> nodeTags;
    std::vector<double> nodeCoords;
    std::vector<double> parametricCoords;
    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, parametricCoords);

    std::vector<int> elementTypes;
    std::vector<std::vector<std::size_t>> elementTags, nodeTagsPerElement;
    gmsh::model::mesh::getElements(elementTypes, elementTags, nodeTagsPerElement);

    std::vector<std::array<double, 3>> points(nodeTags.size());
    for (std::size_t i = 0; i < nodeTags.size(); ++i) {
        points[i] = {nodeCoords[3 * i], nodeCoords[3 * i + 1], nodeCoords[3 * i + 2]};
		// std::cout<<i<<" "<<nodeCoords[3 * i]<<" "<<nodeCoords[3 * i + 1]<<" "<<nodeCoords[3 * i + 2]<<std::endl;
    }
	// std::cout << "points.size() = " << points.size() << std::endl;

	int triID = -1;
	for(int i = 0; i < elementTypes.size(); ++i)
	{
		if(elementTypes[i] == 2)
		{
			triID = i;
			break;
		}
	}
	assert(triID != -1);

    std::vector<std::array<std::size_t, 3>> elements(nodeTagsPerElement[triID].size() / 3);
    for (std::size_t i = 0; i < elements.size(); ++i) {
        elements[i] = {nodeTagsPerElement[triID][3 * i] - 1, nodeTagsPerElement[triID][3 * i + 1] - 1, nodeTagsPerElement[triID][3 * i + 2] - 1};
		// std::cout<<i<<" "<<nodeTagsPerElement[triID][3 * i]<<" "<<nodeTagsPerElement[triID][3 * i + 1]<<" "<<nodeTagsPerElement[triID][3 * i + 2]<<std::endl;
	}

	// for(int i = 0; i < elementTypes.size(); ++i)
	// {
	// 	std::cout << i << " " << elementTypes[i] << std::endl;
	// }
	// std::cout << "nodeTagsPerElement.size() = " << nodeTagsPerElement.size() << std::endl;
	// std::cout << "nodeTagsPerElement[0].size() = " << nodeTagsPerElement[0].size() << std::endl;
	// std::cout << "nodeTagsPerElement[1].size() = " << nodeTagsPerElement[1].size() << std::endl;
	// std::cout << "nodeTagsPerElement[2].size() = " << nodeTagsPerElement[2].size() << std::endl;
	// std::cout << "elements.size() = " << elements.size() << std::endl;
	// std::cout << "elementTags.size() = " << elementTags.size() << std::endl;
	// std::cout << "elementTags[0].size() = " << elementTags[0].size() << std::endl;
	// std::cout << "elementTags[1].size() = " << elementTags[1].size() << std::endl;
	// std::cout << "elementTags[2].size() = " << elementTags[2].size() << std::endl;
	// Finalize Gmsh
	//
	gmsh::finalize();

	// Find boundary nodes
    std::unordered_set<std::size_t> boundaryNodes;
    std::unordered_map<std::size_t, int> nodeCount;

    for (const auto& element : elements) {
        for (const auto& node : element) {
            nodeCount[node]++;
        }
    }

    for (const auto &it : nodeCount)
    {
        const auto &node = it.first;
        const auto &count = it.second;
        // std::cout << node << " node: " << count << std::endl;
        if (count == 3)
        { // Node is on the boundary if it belongs to only one element
            boundaryNodes.insert(node);
            // std::cout << "boundary node: " << node << std::endl;
        }
    }
    std::vector<int> boundaryNodesVec(boundaryNodes.begin(), boundaryNodes.end());

    // Eigen::Matrix3d material_0;
    // material_0 << 1.0, 0.3, 0.0,
    //               0.3, 1.0, 0.0,
    //               0.0, 0.0, 0.5 * (1.0 - 0.3);
    // materials.push_back(material_0);

	Constitutive material_0(10.0, 0.3);
	Constitutive material_1(100.0, 0.3);
	// std::cout<<material_0.get_D()<<std::endl;
	
	// Eigen::Vector3d strain;
    // strain << 0.01, 0.01, 0.0;

    // Eigen::Vector3d stress = material_0.get_stress(strain);

    // std::cout << "Stress: \n" << stress << std::endl;
	
	// Eigen::MatrixXd positions(3, 2);
    // positions << 0.0, 0.0,
    //              0.0, 1.0,
    //              1.0, 0.0;

    // Element element(material_0, positions);

    // std::cout << "Element stiffness matrix:\n" << element.K_element << std::endl;

    // Create the global stiffness matrix
    // GlobalStiffness global_stiffness(points, elements, material_0);
	// std::cout << "global_stiffness.K = " << global_stiffness.K << std::endl;

	// Create the reduced stiffness matrix
    ReducedStiffness reduced_stiffness(points, elements, material_0);

    // Example of setting boundary conditions and forces
	std::unordered_map<int, double> x_fix;
	std::unordered_map<int, double> y_fix;
	std::unordered_map<int, std::array<double, 2>> f_given;

    // for (size_t index = 0; index < points.size(); ++index)
    // {
    //     double x = points[index][0];
    //     double y = points[index][1];
    //     if (x < 1e-6)
    //     {
    //         x_fix[index] = 0.0;
    //         y_fix[index] = 0.0;
    //     }
    //     if ((x - 1.0) < 1e-6 && std::abs(y - 0.0) < 1e-6)
    //     {
    //         f_given[index] = {1.0, 0.0};
    //     }
    // }
    {
        std::array<double, 2> x0 = {0.0, 0.0};
        double radius = 1.0;

        double halfR2 = radius * radius * 0.1;
        for (size_t index = 0; index < reduced_stiffness.nodes.rows(); ++index)
        {
            double x = reduced_stiffness.nodes(index, 0);
            double y = reduced_stiffness.nodes(index, 1);
            if ((x - x0[0]) * (x - x0[0]) + (y - x0[1]) * (y - x0[1]) < halfR2)
            {
                x_fix[index] = 0.0;
                y_fix[index] = 0.0;
            }
        }

        for (int i = 0; i < boundaryNodesVec.size(); ++i)
        {
            int index = boundaryNodesVec[i];

            f_given[index] = {0.0, -0.1};
        }
    }

    reduced_stiffness.export_to_tecplot("circle0.dat");

    reduced_stiffness.apply_conditions(x_fix, y_fix, f_given);

    // Reduce the system
    reduced_stiffness.reduce_system();

    // Solve the reduced system
    reduced_stiffness.solve_reduce_system();

    // Update global variables
    reduced_stiffness.update_global_variables();

    // Update node positions
    reduced_stiffness.update_node_positions();
    // // Define materials

    reduced_stiffness.export_to_tecplot("circle1.dat");

    return 0;
}