#include "small.h"
#include <eigen/Eigen/Dense>
#include <eigen/Eigen/SVD>

using Matrix3d = Eigen::Matrix<SCALAR, 3, 3>;
using Vector3d = Eigen::Matrix<SCALAR, 3, 1>;
using Matrix4d = Eigen::Matrix<SCALAR, 4, 4>;
using Vector4d = Eigen::Matrix<SCALAR, 4, 1>;

extern "C"
{
    void computeInverseWithCheck4d(const SCALAR (*mat)[16], SCALAR (*inv)[16], bool *invertible)
    {
        Eigen::Map<const Matrix4d> m(*mat);
        Eigen::Map<Matrix4d> inverse(*inv);
        m.computeInverseWithCheck(inverse, *invertible);
    }

    void computeInverseWithCheck3d(const SCALAR (*mat)[9], SCALAR (*inv)[9], bool *invertible)
    {
        Eigen::Map<const Matrix3d> m(*mat);
        Eigen::Map<Matrix3d> inverse(*inv);
        m.computeInverseWithCheck(inverse, *invertible);
    }

    // Compute two 3x3 matrices for eigenvectors and eigenvalues.
    // First matrix is a square matrix where each column is an eigenvector.
    // Second matrix is diagonal matrix where each element is an eigenvalue.  
    void computeEigenValuesAndEigenVectors3d(const SCALAR (*mat)[9], SCALAR (*eigenvectors)[9], SCALAR (*eigenvalues)[9])
    {
        Eigen::Map<const Matrix3d> m(*mat);
        Eigen::Map<Matrix3d> vectors(*eigenvectors);
        Eigen::Map<Matrix3d> values(*eigenvalues);
        
        // M is real so we can drop potential imaginary part for eigenvectors and eigenvalues
        Eigen::EigenSolver<Matrix3d> es(m);
        vectors = es.eigenvectors().real();
        values = es.eigenvalues().real().asDiagonal();
    }

    void computeJacobiSVD3d(const SCALAR (*M)[9], SCALAR (*U)[9], SCALAR (*S)[9], SCALAR (*V)[9])
    {
        Eigen::Map<const Matrix3d> m(*M);
        Eigen::Map<Matrix3d> u(*U);
        Eigen::Map<Matrix3d> s(*S);
        Eigen::Map<Matrix3d> v(*V);

        Eigen::JacobiSVD<Matrix3d> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        u = svd.matrixU();
        v = svd.matrixV();
        s = svd.singularValues().asDiagonal();
    }


    void solveSymmetricLinearSystem4d(const SCALAR (*mat)[16], const SCALAR (*b)[4], SCALAR (*x)[4])
    {
        Eigen::Map<const Matrix4d> m(*mat);
        Eigen::Map<const Vector4d> bVec(*b);
        Eigen::Map<Vector4d> xVec(*x);
        Eigen::LDLT<Matrix4d> solver(m);
        xVec = solver.solve(bVec);
    }

    void eigenSolver3d(const SCALAR (*mat)[9], SCALAR (*eigenvalues)[3], SCALAR (*eigenvectors)[9])
    {
        Eigen::Map<const Matrix3d> m(*mat);
        Eigen::Map<Vector3d> evals(*eigenvalues);
        Eigen::Map<Matrix3d> evecs(*eigenvectors);
        Eigen::SelfAdjointEigenSolver<Matrix3d> solver(m);
        evals = solver.eigenvalues();
        evecs = solver.eigenvectors();
    }
}
