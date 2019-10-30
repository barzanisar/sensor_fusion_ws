/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include <unordered_set>
#include "../../processPointClouds.h"
#include "../../render/render.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"

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

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
		new pcl::PointCloud<pcl::PointXYZ>());
	// Add inliers
	float scatter = 0.6;
	for (int i = -5; i < 5; i++)
	{
		double rx = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
		double ry = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
		pcl::PointXYZ point;
		point.x = i + scatter * rx;
		point.y = i + scatter * ry;
		point.z = 0;

		cloud->points.push_back(point);
	}
	// Add outliers
	int numOutliers = 10;
	while (numOutliers--)
	{
		double rx = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
		double ry = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
		pcl::PointXYZ point;
		point.x = 5 * rx;
		point.y = 5 * ry;
		point.z = 0;

		cloud->points.push_back(point);
	}
	cloud->width = cloud->points.size();
	cloud->height = 1;

	return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}

pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(
		new pcl::visualization::PCLVisualizer("2D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->initCameraParameters();
	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
	viewer->addCoordinateSystem(1.0);
	return viewer;
}

std::unordered_set<int> RansacLine(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
								   int maxIterations, float distanceTol)
{
	std::unordered_set<int> inliersResult;
	srand(time(NULL));

	// My soln:
	// std::unordered_set<int> inliersForThisIter;
	// int index1, index2;
	// pcl::PointXYZ p1, p2;
	// float A, B, C, pointDist;
	// int maxInlierCount = 0;

	// for (int iter = 0; iter < maxIterations; ++iter)
	// {
	// 	index1 = rand() % cloud->points.size();
	// 	index2 = rand() % cloud->points.size();

	// 	while (index1 == index2)
	// 	{
	// 		index2 = rand() % cloud->points.size();
	// 	}

	// 	p1 = cloud->points[index1];
	// 	p2 = cloud->points[index2];

	// 	// Fit line: Find line coefficients
	// 	A = p1.y - p2.y;
	// 	B = p2.x - p1.x;
	// 	C = p1.x * p2.y - p2.x * p1.y;

	// 	inliersForThisIter.clear();
	// 	for (int index = 0; index < cloud->points.size(); index++)
	// 	{
	// 		pcl::PointXYZ point = cloud->points[index];
	// 		pointDist = fabs(A*point.x + B*point.y + C) / sqrt(A*A + B*B);

	// 		if (pointDist <= distanceTol)
	// 		{
	// 			inliersForThisIter.insert(index);
	// 		}
	// 	}

	// 	if (inliersForThisIter.size() > maxInlierCount)
	// 	{
	// 		maxInlierCount = inliersForThisIter.size();
	// 		inliersResult = inliersForThisIter;
	// 	}

	// }

	while (maxIterations--)
	{
		std::unordered_set<int> inliersForThisIter;

		// Pick 2 random points as inliers. Make sure they are unique.
		while (inliersForThisIter.size() < 2)
		{
			inliersForThisIter.insert(rand() % (cloud->points.size()));
		}

		auto itr = inliersForThisIter.begin();
		float x1, y1, x2, y2;
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;

		// Fit a line through these 2 points.
		float a = y1 - y2;
		float b = x2 - x1;
		float c = x1 * y2 - x2 * y1;

		// Loop over remaining points in the cloud and add them if inliers
		for (int index = 0; index < cloud->points.size(); index++)
		{
			// if inlier already exists i.e. used to fit line.
			// Don't compute its distance to line.
			if (inliersForThisIter.count(index))
				continue;

			pcl::PointXYZ point = cloud->points[index];
			float pointDist =
				fabs(a * point.x + b * point.y + c) / sqrt(a * a + b * b);

			if (pointDist <= distanceTol)
			{
				inliersForThisIter.insert(index);
			}
		}

		if (inliersForThisIter.size() > inliersResult.size())
		{
			inliersResult = inliersForThisIter;
		}
	}

	return inliersResult;
}

std::unordered_set<int> RansacPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
									int maxIterations, float distanceTol)
{
	// Time RANSAC
	auto startTime = std::chrono::steady_clock::now();

	std::unordered_set<int> inliersResult;
	srand(time(NULL));

	while (maxIterations--)
	{
		std::unordered_set<int> inliersForThisIter;

		// Pick 3 random points as inliers. Make sure they are unique.
		while (inliersForThisIter.size() < 3)
		{
			inliersForThisIter.insert(rand() % (cloud->points.size()));
		}

		std::vector<Vect3f> pointsOnPlane;
		for (auto const &index : inliersForThisIter)
		{
			pointsOnPlane.push_back(Vect3f(cloud->points[index].x,
										   cloud->points[index].y,
										   cloud->points[index].z));
		}

		// for (auto it = inliersForThisIter.begin(); it !=
		// inliersForThisIter.end(); ++it)
		// {
		// 	pointsOnPlane.push_back(Vect3(cloud->points[*it].x,
		// 							 cloud->points[*it].y,
		// 							 cloud->points[*it].z));
		// }

		// for (std::unordered_set<int>::iterator it = inliersForThisIter.begin();
		// it != inliersForThisIter.end(); ++it)
		// {
		// 	pointsOnPlane.push_back(Vect3(cloud->points[*it].x,
		// 							 cloud->points[*it].y,
		// 							 cloud->points[*it].z));
		//	}

		// Fit a plane through these 3 points.
		Vect3f v1 = pointsOnPlane[1] - pointsOnPlane[0];
		Vect3f v2 = pointsOnPlane[2] - pointsOnPlane[0];
		Vect3f planeNormal = v1.cross(v2);
		float a, b, c, d;
		a = planeNormal.x;
		b = planeNormal.y;
		c = planeNormal.z;
		d = -(a * pointsOnPlane[0].x + b * pointsOnPlane[0].y +
			  c * pointsOnPlane[0].z);

		// Loop over remaining points in the cloud and add them if inliers
		for (int index = 0; index < cloud->points.size(); index++)
		{
			// if inlier already exists i.e. used to fit line.
			// Don't compute its distance to line.
			if (inliersForThisIter.count(index))
				continue;

			pcl::PointXYZ point = cloud->points[index];
			float pointDist = fabs(a * point.x + b * point.y + c * point.z + d) /
							  sqrt(a * a + b * b + c * c);

			if (pointDist <= distanceTol)
			{
				inliersForThisIter.insert(index);
			}
		}

		if (inliersForThisIter.size() > inliersResult.size())
		{
			inliersResult = inliersForThisIter;
		}
	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
		endTime - startTime);
	std::cout << "RANSAC took " << elapsedTime.count() << " milliseconds"
			  << std::endl;

	return inliersResult;
}
int main()
{
	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

	// Create data
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();

	// TODO: Change the max iteration and distance tolerance arguments for Ransac
	// function
	std::unordered_set<int> inliers = RansacPlane(cloud, 100, 0.2);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInliers(
		new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(
		new pcl::PointCloud<pcl::PointXYZ>());

	for (int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if (inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}

	// Render 2D point cloud with inliers and outliers
	if (inliers.size())
	{
		renderPointCloud(viewer, cloudInliers, "inliers", Color(0, 1, 0));
		renderPointCloud(viewer, cloudOutliers, "outliers", Color(1, 0, 0));
	}
	else
	{
		renderPointCloud(viewer, cloud, "data");
	}

	while (!viewer->wasStopped())
	{
		viewer->spinOnce();
	}
}
