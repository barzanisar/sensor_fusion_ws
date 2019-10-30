/* \author Aaron Brown */
// Quiz on implementing kd tree

#include "../render/render.h"

// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	uint id;
	Node *left;
	Node *right;

	Node(const std::vector<float> &arr, const uint &setId)
		: point(arr), id(setId), left(NULL), right(NULL)
	{
	}
};

struct KdTree
{
	Node *root;
	uint _pointDimension;

	KdTree(const uint &dimension)
		: root(NULL), _pointDimension(dimension)
	{
	}

	void insertHelper(Node *&node, const uint &depth, const std::vector<float> &point, const uint &id)
	{
		// Tree is empty
		if (node == NULL)
		{
			node = new Node(point, id);
		}
		else
		{
			// Calculate dimension to compare
			uint curr_dim = depth % _pointDimension;

			if (point[curr_dim] < node->point[curr_dim])
			{
				insertHelper(node->left, depth + 1, point, id);
			}
			else
			{
				insertHelper(node->right, depth + 1, point, id);
			}
		}
	}
	void insert(const std::vector<float> &point, const uint &id)
	{
		// Insert a new point into the tree
		// the function should create a new node and place correctly with in the root
		insertHelper(root, 0, point, id);
	}

	void searchHelper(const std::vector<float> &target, const Node *const node, const uint &depth, const float &distanceTol, std::vector<uint> &ids)
	{
		if (node != NULL)
		{
			bool isNearby = true;

			for (uint dim = 0; dim < _pointDimension; ++dim)
			{
				isNearby = isNearby && node->point[dim] >= (target[dim] - distanceTol) && node->point[dim] <= (target[dim] + distanceTol);
			}

			if (isNearby)
			{
				float distance = 0.0;
				for (uint dim = 0; dim < _pointDimension; ++dim)
				{
					distance += ((node->point[dim] - target[dim]) * (node->point[dim] - target[dim]));
				}
				distance = sqrt(distance);

				if (distance <= distanceTol)
				{
					ids.push_back(node->id);
				}
			}

			uint curr_dim = depth % _pointDimension;

			if ((target[curr_dim] - distanceTol) < node->point[curr_dim])
			{
				searchHelper(target, node->left, depth + 1, distanceTol, ids);
			}
			if ((target[curr_dim] + distanceTol) > node->point[curr_dim])
			{
				searchHelper(target, node->right, depth + 1, distanceTol, ids);
			}
		}
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<uint> search(const std::vector<float> &target, const float &distanceTol)
	{
		std::vector<uint> ids;
		searchHelper(target, root, 0, distanceTol, ids);
		return ids;
	}
};
