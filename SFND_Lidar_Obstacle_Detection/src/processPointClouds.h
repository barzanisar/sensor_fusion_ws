// PCL lib Functions for processing point clouds

#ifndef PROCESSPOINTCLOUDS_H_
#define PROCESSPOINTCLOUDS_H_

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <ctime>
#include <chrono>
#include "render/box.h"
#include "cluster/kdtree.h"

struct Vect3f
{
    float x, y, z;

    Vect3f(float setX, float setY, float setZ) : x(setX), y(setY), z(setZ) {}

    Vect3f operator+(const Vect3f &vecRhs)
    {
        Vect3f result(x + vecRhs.x, y + vecRhs.y, z + vecRhs.z);
        return result;
    }

    Vect3f operator-(const Vect3f &vecRhs)
    {
        Vect3f result(x - vecRhs.x, y - vecRhs.y, z - vecRhs.z);
        return result;
    }

    Vect3f cross(const Vect3f &vecRhs)
    {
        Vect3f result(y * vecRhs.z - z * vecRhs.y, z * vecRhs.x - x * vecRhs.z,
                      x * vecRhs.y - y * vecRhs.x);
        return result;
    }
};

template <typename PointT>
class ProcessPointClouds
{
    void fillCluster(typename pcl::PointCloud<PointT>::Ptr cloud, const uint &index, KdTree *const &tree, const float &distanceTol, std::vector<bool> &processed, std::vector<uint> &cluster);

public:
    //constructor
    ProcessPointClouds();
    //deconstructor
    ~ProcessPointClouds();

    void numPoints(typename pcl::PointCloud<PointT>::Ptr cloud);

    typename pcl::PointCloud<PointT>::Ptr FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold);

    std::vector<typename pcl::PointCloud<PointT>::Ptr> Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize);

    Box BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster);

    void savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file);

    typename pcl::PointCloud<PointT>::Ptr loadPcd(std::string file);

    std::vector<boost::filesystem::path> streamPcd(std::string dataPath);
};
#endif /* PROCESSPOINTCLOUDS_H_ */