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

template <typename PointT>
class ProcessPointClouds
{
public:
    //constructor
    ProcessPointClouds();
    //deconstructor
    ~ProcessPointClouds();

    void numPoints(const typename pcl::PointCloud<PointT>::Ptr cloud);

    typename pcl::PointCloud<PointT>::Ptr FilterCloud(const typename pcl::PointCloud<PointT>::Ptr cloud, const float filterRes, const Eigen::Vector4f &minPoint, const Eigen::Vector4f &maxPoint);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SeparateClouds(const pcl::PointIndices::Ptr inliers, const typename pcl::PointCloud<PointT>::Ptr cloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SegmentPlane(const typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, const float distanceThreshold);

    std::vector<typename pcl::PointCloud<PointT>::Ptr> Clustering(const typename pcl::PointCloud<PointT>::Ptr cloud, const float clusterTolerance, const int minSize, const int maxSize);

    Box BoundingBox(const typename pcl::PointCloud<PointT>::Ptr cluster);

    void savePcd(const typename pcl::PointCloud<PointT>::Ptr cloud, const std::string &file);

    typename pcl::PointCloud<PointT>::Ptr loadPcd(const std::string &file);

    std::vector<boost::filesystem::path> streamPcd(const std::string &dataPath);

private:
    void FillCluster(typename pcl::PointCloud<PointT>::Ptr cloud, const uint index, KdTree *const &tree, const float distanceTol, std::vector<bool> &processed, typename pcl::PointCloud<PointT>::Ptr cluster);
};
#endif /* PROCESSPOINTCLOUDS_H_ */