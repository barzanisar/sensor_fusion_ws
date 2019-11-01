// PCL lib Functions for processing point clouds

#include "processPointClouds.h"

//constructor:
template <typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}

//de-constructor:
template <typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}

template <typename PointT>
void ProcessPointClouds<PointT>::numPoints(const typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(const typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, const Eigen::Vector4f &minPoint, const Eigen::Vector4f &maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    typename pcl::PointCloud<PointT>::Ptr cloudFiltered(new pcl::PointCloud<PointT>);

    // Extract region of interest (region based filtering)
    pcl::CropBox<PointT> regionOfInterest(true);
    regionOfInterest.setMin(minPoint);
    regionOfInterest.setMax(maxPoint);
    regionOfInterest.setInputCloud(cloud);
    regionOfInterest.filter(*cloudFiltered);

    // Downsample the region of interest using voxel grid point reduction.
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloudFiltered);
    vg.setLeafSize(filterRes, filterRes, filterRes);
    vg.filter(*cloudFiltered);

    // Remove the car roof points.
    pcl::CropBox<PointT> cropRoof(true);
    // Set the min and max points of the roof region to crop/remove.
    cropRoof.setMin(Eigen::Vector4f(-1.5, -1.7, -1, 1));
    cropRoof.setMax(Eigen::Vector4f(2.6, 2, -0.4, 1));
    cropRoof.setInputCloud(cloudFiltered);
    cropRoof.setNegative(true);
    cropRoof.filter(*cloudFiltered);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudFiltered;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(const pcl::PointIndices::Ptr inliers, const typename pcl::PointCloud<PointT>::Ptr cloud)
{
    // Create two new point clouds, one cloud with obstacles and other with segmented plane
    typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr obsCloud(new pcl::PointCloud<PointT>());

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);

    // Method 1: Extract plane cloud
    // extract.setNegative(false);
    // extract.filter(*planeCloud);

    // Method 2: Extract plane Cloud
    for (int inlierIndex : inliers->indices)
    {
        planeCloud->points.push_back(cloud->points[inlierIndex]);
    }

    // Extract obstacle cloud by subtracting plane inliers from cloud
    extract.setNegative(true);
    extract.filter(*obsCloud);
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obsCloud, planeCloud);
    return segResult;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(const typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, const float distanceThreshold)
{
    // ------------Method 1: My RANSAC-------------
    // --------------------------------------------

    // Time RANSAC
    auto startTime = std::chrono::steady_clock::now();

    std::unordered_set<int> inliersSet;

    srand(time(NULL));

    while (maxIterations--)
    {
        std::unordered_set<int> inliersForThisIter;

        // Pick 3 random points as inliers. Make sure they are unique.
        while (inliersForThisIter.size() < 3)
        {
            inliersForThisIter.insert(rand() % (cloud->points.size()));
        }

        std::vector<Eigen::Vector3f> pointsOnPlane;
        for (auto index : inliersForThisIter)
        {
            pointsOnPlane.push_back(Eigen::Vector3f(cloud->points[index].x,
                                                    cloud->points[index].y,
                                                    cloud->points[index].z));
        }

        // Fit a plane through these 3 points.
        Eigen::Vector3f v1 = pointsOnPlane[1] - pointsOnPlane[0];
        Eigen::Vector3f v2 = pointsOnPlane[2] - pointsOnPlane[0];
        Eigen::Vector3f planeNormal = v1.cross(v2);
        float a, b, c, d;
        a = planeNormal.x();
        b = planeNormal.y();
        c = planeNormal.z();
        d = -(a * pointsOnPlane[0].x() + b * pointsOnPlane[0].y() +
              c * pointsOnPlane[0].z());

        // Loop over remaining points in the cloud and add them if inliers
        for (int index = 0; index < cloud->points.size(); index++)
        {
            // If inlier already exists i.e. used to fit line.
            // Don't compute its distance to line.
            if (inliersForThisIter.count(index))
                continue;

            PointT point = cloud->points[index];
            float pointDistToPlane = fabs(a * point.x + b * point.y + c * point.z + d) /
                                     sqrt(a * a + b * b + c * c);

            if (pointDistToPlane <= distanceThreshold)
            {
                inliersForThisIter.insert(index);
            }
        }

        if (inliersForThisIter.size() > inliersSet.size())
        {
            inliersSet = inliersForThisIter;
        }
    }

    // Copy unordered set of indices to pcl::PointIndices
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    std::copy(inliersSet.begin(),
              inliersSet.end(),
              std::inserter(inliers->indices, inliers->indices.end()));

    // Another method for copying
    // for (auto index : inliersSet)
    // {
    //     inliers->indices.push_back(index);
    // }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);
    std::cout << "RANSAC took " << elapsedTime.count() << " milliseconds"
              << std::endl;

    // ------------Method 2: PCL RANSAC-------------
    // --------------------------------------------

    // // Time PCL segmentation process
    // auto startTime = std::chrono::steady_clock::now();

    // pcl::SACSegmentation<PointT> seg;
    // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    // // Optional
    // seg.setOptimizeCoefficients(true);
    // // Mandatory
    // seg.setModelType(pcl::SACMODEL_PLANE);
    // seg.setMethodType(pcl::SAC_RANSAC);
    // seg.setMaxIterations(maxIterations);
    // seg.setDistanceThreshold(distanceThreshold);

    // // Segment the largest planar component from the remaining cloud
    // seg.setInputCloud(cloud);
    // seg.segment(*inliers, *coefficients);
    // if (inliers->indices.size() == 0)
    // {
    //     std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    // }

    // auto endTime = std::chrono::steady_clock::now();
    // auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    // std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, cloud);
    return segResult;
}

template <typename PointT>
void ProcessPointClouds<PointT>::fillCluster(typename pcl::PointCloud<PointT>::Ptr cloud, const uint index, KdTree *const &tree, const float distanceTol, std::vector<bool> &processed, std::vector<uint> &cluster)
{
    processed[index] = true;
    cluster.push_back(index);
    std::vector<uint> nearby = tree->search({cloud->points[index].x, cloud->points[index].y, cloud->points[index].z}, distanceTol);
    for (uint idx : nearby)
    {
        if (!processed[idx])
            fillCluster(cloud, idx, tree, distanceTol, processed, cluster);
    }
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(const typename pcl::PointCloud<PointT>::Ptr cloud, const float clusterTolerance, const int minSize, const int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    // ------------Method 1: My Clustering--------
    // --------------------------------------------

    // Construct 3d-Tree for the cloud
    KdTree *tree = new KdTree(3);

    for (uint i = 0; i < cloud->points.size(); i++)
        tree->insert({cloud->points[i].x, cloud->points[i].y, cloud->points[i].z}, i);

    // Find clusters
    std::vector<std::vector<uint>> clusters;

    std::vector<bool> processed(cloud->points.size(), false);

    for (uint index = 0; index < cloud->points.size(); ++index)
    {
        if (!processed[index])
        {
            std::vector<uint> cluster;
            fillCluster(cloud, index, tree, clusterTolerance, processed, cluster);
            if (cluster.size() >= minSize && cluster.size() <= maxSize)
                clusters.push_back(cluster);
        }
    }

    // Convert clusters of indices to clouds
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusterClouds;
    clusterClouds.resize(clusters.size());

    for (uint cluster_id = 0; cluster_id < clusters.size(); ++cluster_id)
    {
        typename pcl::PointCloud<PointT>::Ptr clusterCloud(new pcl::PointCloud<PointT>);
        for (uint index : clusters[cluster_id])
        {
            clusterCloud->points.push_back(cloud->points[index]);
        }
        clusterCloud->width = clusterCloud->points.size();
        clusterCloud->height = 1;
        clusterCloud->is_dense = true;

        clusterClouds[cluster_id] = clusterCloud;
    }

    // ------------Method 2: PCL Clustering--------
    // --------------------------------------------

    // std::vector<typename pcl::PointCloud<PointT>::Ptr> clusterClouds;

    // // Perform euclidean clustering to group detected obstacles

    // // Creating the KdTree object for the search method of the extraction
    // typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    // tree->setInputCloud(cloud);

    // std::vector<pcl::PointIndices> cluster_indices;
    // pcl::EuclideanClusterExtraction<PointT> ec;
    // ec.setClusterTolerance(clusterTolerance);
    // ec.setMinClusterSize(minSize);
    // ec.setMaxClusterSize(maxSize);
    // ec.setSearchMethod(tree);
    // ec.setInputCloud(cloud);
    // ec.extract(cluster_indices);

    // clusterClouds.resize(cluster_indices.size());

    // for (int cluster_id = 0; cluster_id < cluster_indices.size(); ++cluster_id)
    // {
    //     typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>);
    //     for (int index : cluster_indices[cluster_id].indices)
    //     {
    //         cloudCluster->points.push_back(cloud->points[index]);
    //     }
    //     cloudCluster->width = cloudCluster->points.size();
    //     cloudCluster->height = 1;
    //     cloudCluster->is_dense = true;

    //     clusterClouds[cluster_id] = cloudCluster;
    // }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusterClouds.size() << " clusters" << std::endl;

    return clusterClouds;
}

template <typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(const typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

template <typename PointT>
void ProcessPointClouds<PointT>::savePcd(const typename pcl::PointCloud<PointT>::Ptr cloud, const std::string &file)
{
    pcl::io::savePCDFileASCII(file, *cloud);
    std::cerr << "Saved " << cloud->points.size() << " data points to " + file << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(const std::string &file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size() << " data points from " + file << std::endl;

    return cloud;
}

template <typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(const std::string &dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;
}