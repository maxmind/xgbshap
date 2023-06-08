package xgbshap

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// XXX Some of this code is similar to parse.go in xgb2code
// (https://github.com/maxmind/xgb2code).

// XGBModel corresponds to an XGBoost JSON model.
type XGBModel struct {
	Learner Learner `json:"learner"`
}

// Learner is the top level part of an XGBoost model.
type Learner struct {
	Attributes      Attributes      `json:"attributes"`
	GradientBooster GradientBooster `json:"gradient_booster"`
}

// Attributes holds attributes from an XGBoost model.
type Attributes struct {
	BestNtreeLimit json.Number `json:"best_ntree_limit"`
}

// GradientBooster holds the XGBoost model.
type GradientBooster struct {
	Model Model `json:"model"`
}

// Model is the XGBoost model.
type Model struct {
	Trees []XGBTree `json:"trees"`
}

// XGBTree is one tree in an XGBoost model as decoded from JSON.
type XGBTree struct {
	BaseWeights     []float32 `json:"base_weights"`
	DefaultLeft     []int     `json:"default_left"`
	LeftChildren    []int     `json:"left_children"`
	RightChildren   []int     `json:"right_children"`
	SplitConditions []float32 `json:"split_conditions"`
	SplitIndices    []int     `json:"split_indices"`
	SumHessian      []float32 `json:"sum_hessian"`
	TreeParam       TreeParam `json:"tree_param"`
}

// TreeParam holds tree parameters.
type TreeParam struct {
	NumNodes json.Number `json:"num_nodes"`
}

// Tree is one tree in an XGBoost model. It's the representation we process
// XGBTree into.
type Tree struct {
	Nodes    []*Node // Index 0 is the root.
	NumNodes int
}

// Node is a node in the Tree.
type Node struct {
	Data  NodeData
	Left  *Node
	Right *Node
}

// NodeData is a Node's data.
type NodeData struct {
	BaseWeight     float32
	DefaultLeft    bool
	ID             int
	SplitCondition float32
	SplitIndex     int
	SumHessian     float32
}

// IsLeaf returns whether the Node is a leaf.
//
// This is equivalent to IsLeaf() in xgboost (tree_model.h).
func (n *Node) IsLeaf() bool { return n.Left == nil }

// LeafValue returns the leaf's value.
//
// This is equivalent to LeafValue() in xgboost (tree_model.h).
func (n *Node) LeafValue() float32 { return n.Data.BaseWeight }

// MaxDepth returns the tree's max depth at this node.
//
// This is equivalent to MaxDepth() in xgboost (tree_model.h).
func (n *Node) MaxDepth() int {
	if n.IsLeaf() {
		return 0
	}

	leftDepth := n.Left.MaxDepth() + 1
	rightDepth := n.Right.MaxDepth() + 1

	return int(math.Max(float64(leftDepth), float64(rightDepth)))
}

func parseModel(
	file string,
) (*XGBModel, []*Tree, error) {
	buf, err := os.ReadFile(filepath.Clean(file))
	if err != nil {
		return nil, nil, fmt.Errorf("reading file: %w", err)
	}

	var xm XGBModel
	if err := json.Unmarshal(buf, &xm); err != nil {
		return nil, nil, fmt.Errorf("unmarshaling: %w", err)
	}

	var trees []*Tree
	//nolint:gocritic // Copies inefficiently, but should only be done once.
	for _, t := range xm.Learner.GradientBooster.Model.Trees {
		tree, err := parseTree(t)
		if err != nil {
			return nil, nil, err
		}

		trees = append(trees, tree)
	}

	return &xm, trees, nil
}

func parseTree(
	xt XGBTree,
) (*Tree, error) {
	numNodes, err := xt.TreeParam.NumNodes.Int64()
	if err != nil {
		return nil, fmt.Errorf("getting num nodes as int64: %w", err)
	}

	var nodes []*Node
	for i := 0; i < int(numNodes); i++ {
		nodes = append(nodes, &Node{})
	}

	for i := 0; i < int(numNodes); i++ {
		nodes[i].Data = NodeData{
			BaseWeight:     xt.BaseWeights[i],
			DefaultLeft:    xt.DefaultLeft[i] == 1,
			ID:             i,
			SplitCondition: xt.SplitConditions[i],
			SplitIndex:     xt.SplitIndices[i],
			SumHessian:     xt.SumHessian[i],
		}

		left := xt.LeftChildren[i]
		right := xt.RightChildren[i]

		if left == -1 { // No child
			continue
		}

		nodes[i].Left = nodes[left]
		nodes[i].Right = nodes[right]
	}

	return &Tree{
		Nodes:    nodes,
		NumNodes: int(numNodes),
	}, nil
}
