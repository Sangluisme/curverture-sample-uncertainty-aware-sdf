#include "MarchingCubes.h"
#include "MarchingCubesNoColor.h"
#include "mat.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <stdexcept>
#include <array>

#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"
#include <pybind11/eigen.h>

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;



std::tuple<py::array_t<float>, py::array_t<float>> make_tuple(std::vector<Vec3f>& vec1, std::vector<Vec3i>& vec2) {

   
    // Create NumPy arrays from the two matrices
    Eigen::MatrixXf mat1(vec1.size(), 3);

    // Map the data in the vector to the matrix
    Eigen::Map<Eigen::MatrixXf>(mat1.data(), mat1.rows(), mat1.cols()) = Eigen::Map<const Eigen::MatrixXf>(vec1[0].data(), vec1.size(), 3);

    Eigen::MatrixXi mat2(vec2.size(), 3);

    Eigen::Map<Eigen::MatrixXi>(mat2.data(), mat2.rows(), mat2.cols()) = Eigen::Map<const Eigen::MatrixXi>(vec2[0].data(), vec2.size(), 3);
    
    
    
    // Return a Python tuple of the two NumPy arrays
    return std::make_tuple(py::cast(mat1), py::cast(mat2));
}


PYBIND11_MODULE(weight_marching_cubes, m) {
    m.doc() = "MarchingCubes wrapper";

    py::class_<MarchingCubes>(m, "weight_marching_cubes")
    .def(py::init<const Vec3i&, const Vec3f&, const Vec3f&>())
    .def("savePly", &MarchingCubes::savePly)  
    // .def("computeIsoSurface", &MarchingCubes::computeIsoSurface, py::arg("tsdf"), py::arg("weight"), py::arg("red"),py::arg("green"), py::arg("blue"), py::arg("isolevel") )
    .def("computeIsoSurface", &MarchingCubes::computeIsoSurface);
    m.def("MarchingCubes",[](py::array_t<int> grid_dim, py::array_t<float> size, py::array_t<float> origin){
        int* dim = grid_dim.mutable_data();
        float* voxel_size = size.mutable_data();
        float* shift = origin.mutable_data();
        Vec3i dim_ = Eigen::Map<Vec3i>(dim, grid_dim.size());
        Vec3f size_ = Eigen::Map<Vec3f>(voxel_size, 3);
        Vec3f origin_ = Eigen::Map<Vec3f>(shift, 3);

        return new MarchingCubes(dim_, size_, origin_);

    });

    m.def("computeIsoSurface", [](MarchingCubes& mc, py::array_t<float> volume, py::array_t<float> w, py::array_t<unsigned char> red, py::array_t<unsigned char> green, py::array_t<unsigned char> blue, float iso_level=0.0f){
        float* tsdf = volume.mutable_data();
        float* weight = w.mutable_data();
        unsigned char* r = static_cast<unsigned char*>(red.mutable_data());
        unsigned char* g = static_cast<unsigned char*>(green.mutable_data());
        unsigned char* b = static_cast<unsigned char*>(blue.mutable_data());

        mc.computeIsoSurface(tsdf, weight, r, g, b, iso_level);
        std::vector<Vec3f> vertices = mc.get_vertices();
        std::vector<Vec3i> faces = mc.get_faces();

        return make_tuple(vertices, faces);


    });

    m.def("savePly", [](MarchingCubes& mc, const std::string& filename){
        return mc.savePly(filename);
    });
    
}