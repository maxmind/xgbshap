package xgbshap

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
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
	BestIteration  json.Number `json:"best_iteration"`
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
	SplitConditions []xgbFloat `json:"split_conditions"`
	SplitIndices    []int     `json:"split_indices"`
	SumHessian      []float32 `json:"sum_hessian"`
	TreeParam       TreeParam `json:"tree_param"`
	// SplitType marks each node's split kind: 0 = numeric, 1 = categorical. It
	// is absent in models trained without categorical features, in which case
	// every split is numeric.
	SplitType []int `json:"split_type"`
	// The fields below describe categorical splits. Categories is the flattened
	// list of category values across all categorical nodes; for each entry k in
	// CategoriesNodes (a node ID), the values that route to that node's right
	// child are Categories[CategoriesSegments[k] : CategoriesSegments[k]+CategoriesSizes[k]].
	Categories         []int `json:"categories"`
	CategoriesNodes    []int `json:"categories_nodes"`
	CategoriesSegments []int `json:"categories_segments"`
	CategoriesSizes    []int `json:"categories_sizes"`
}

// TreeParam holds tree parameters.
type TreeParam struct {
	NumNodes json.Number `json:"num_nodes"`
}

// xgbFloat is a float32 decoded from XGBoost's JSON, where a number may appear
// either as a normal JSON number or as one of the non-finite tokens Infinity,
// -Infinity, and NaN that XGBoost emits but standard JSON forbids. parseModel
// rewrites those tokens to quoted strings before decoding (see
// sanitizeNonFiniteNumbers), so this unmarshaler accepts a JSON number or any
// quoted float literal (which includes the rewritten tokens). It is currently
// used only for the split_conditions field, the one place these tokens occur.
type xgbFloat float32

func (s *xgbFloat) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '"' {
		var str string
		if err := json.Unmarshal(b, &str); err != nil {
			return fmt.Errorf("decoding split_condition string: %w", err)
		}
		f, err := strconv.ParseFloat(str, 32)
		if err != nil {
			return fmt.Errorf("invalid split_condition %q: %w", str, err)
		}
		*s = xgbFloat(f)
		return nil
	}
	var f float32
	if err := json.Unmarshal(b, &f); err != nil {
		return fmt.Errorf("decoding split_condition number: %w", err)
	}
	*s = xgbFloat(f)
	return nil
}

// Tree is one tree in an XGBoost model. It's the representation we process
// XGBTree into.
type Tree struct {
	Nodes    []Node // Index 0 is the root.
	NumNodes int
}

// Node is a node in the Tree.
type Node struct {
	Left  *Node
	Right *Node
	Data  NodeData
}

// NodeData is a Node's data.
type NodeData struct {
	ID             int
	SplitIndex     int
	SplitCondition float32
	SumHessian     float32
	BaseWeight     float32
	DefaultLeft    bool
	// Categorical reports whether this is a categorical split. When true,
	// Categories holds the category values that route to the right child and
	// SplitCondition is unused (XGBoost stores a dummy threshold there).
	Categorical bool
	Categories  []int
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

	return max(leftDepth, rightDepth)
}

// sanitizeNonFiniteNumbers rewrites the JSON-incompatible literals XGBoost emits
// for non-finite floats (Infinity, -Infinity, NaN) into quoted strings, anywhere
// they appear outside a JSON string literal (never inside string contents). In
// well-formed XGBoost output these tokens only ever appear as numeric values.
// encoding/json rejects these tokens at the lexer level, before any custom
// unmarshaler can see them, so they must be rewritten in the raw bytes;
// xgbFloat.UnmarshalJSON then accepts the quoted form. The input is returned
// unchanged when no such token is present.
//
// This is the same logic as sanitizeNonFiniteNumbers in xgb2code's parse.go.
func sanitizeNonFiniteNumbers(data []byte) []byte {
	// "Infinity" is a substring of "-Infinity", so this also detects the latter.
	if !bytes.Contains(data, []byte("Infinity")) &&
		!bytes.Contains(data, []byte("NaN")) {
		return data
	}

	// Checked longest-first so -Infinity is matched before Infinity.
	tokens := [][]byte{[]byte("-Infinity"), []byte("Infinity"), []byte("NaN")}

	out := make([]byte, 0, len(data))
	inString := false
	for i := 0; i < len(data); {
		c := data[i]
		if inString {
			out = append(out, c)
			// Skip the escaped character so an escaped quote does not end the
			// string prematurely.
			if c == '\\' && i+1 < len(data) {
				out = append(out, data[i+1])
				i += 2
				continue
			}
			if c == '"' {
				inString = false
			}
			i++
			continue
		}
		if c == '"' {
			inString = true
			out = append(out, c)
			i++
			continue
		}
		matched := false
		for _, tok := range tokens {
			if !bytes.HasPrefix(data[i:], tok) {
				continue
			}
			out = append(out, '"')
			out = append(out, tok...)
			out = append(out, '"')
			i += len(tok)
			matched = true
			break
		}
		if matched {
			continue
		}
		out = append(out, c)
		i++
	}
	return out
}

// checkSplitCondition validates a node's split_conditions value. The value is a
// numeric threshold for a numeric decision node, a leaf's output value for a
// leaf, and a dummy (ignored) value for a categorical node. A -Infinity
// threshold on a numeric decision node is accepted: it represents a
// "missingness split" from XGBoost's histogram-based split finder that routes
// purely on whether the feature is missing. All other non-finite values are
// rejected because they have no well-defined routing semantics.
func checkSplitCondition(id int, sc float32, categorical, isLeaf bool) error {
	if isLeaf {
		return nil
	}
	if categorical {
		// The value is a dummy for categorical nodes and is never used.
		return nil
	}
	if math.IsInf(float64(sc), -1) {
		return nil
	}
	if math.IsInf(float64(sc), 0) || math.IsNaN(float64(sc)) {
		return fmt.Errorf(
			"node %d has non-finite split threshold (%v); only finite values "+
				"and -Infinity are supported",
			id,
			sc,
		)
	}
	return nil
}

// categorySets maps each categorical node's ID to the category values that
// route to its right child, decoding XGBoost's flattened categories/segments/
// sizes representation. It returns an empty map for models trained without
// categorical features. It validates the arrays rather than trusting them: a
// malformed or inconsistent encoding would otherwise cause a categorical node
// to be silently treated as a numeric split on its dummy threshold, producing
// wrong contributions.
//
// This is the same logic as categorySets in xgb2code's parse.go.
func categorySets(xt XGBTree, numNodes int64) (map[int][]int, error) {
	n := len(xt.CategoriesNodes)
	if len(xt.CategoriesSegments) != n || len(xt.CategoriesSizes) != n {
		return nil, fmt.Errorf(
			"inconsistent categorical arrays: categories_nodes=%d, "+
				"categories_segments=%d, categories_sizes=%d",
			n,
			len(xt.CategoriesSegments),
			len(xt.CategoriesSizes),
		)
	}

	sets := make(map[int][]int, n)
	for k := range n {
		start := xt.CategoriesSegments[k]
		size := xt.CategoriesSizes[k]
		if start < 0 || size < 0 || start > len(xt.Categories)-size {
			return nil, fmt.Errorf(
				"categorical segment [%d:%d+%d] out of range for "+
					"categories of length %d",
				start,
				start,
				size,
				len(xt.Categories),
			)
		}
		nodeID := xt.CategoriesNodes[k]
		if nodeID < 0 || int64(nodeID) >= numNodes {
			return nil, fmt.Errorf(
				"categories_nodes[%d] = %d out of range for num_nodes %d",
				k,
				nodeID,
				numNodes,
			)
		}
		cats := make([]int, size)
		copy(cats, xt.Categories[start:start+size])
		sets[nodeID] = cats
	}

	// split_type is the only independent signal of which nodes are categorical,
	// so it is what lets us verify that every categorical node was decoded.
	// Without it we cannot make that check, and a categorical node missing from
	// categories_nodes would be silently treated as a numeric split on its dummy
	// threshold. Real XGBoost models always include split_type when they have
	// categorical data, so reject categorical data that lacks it rather than
	// risk wrong contributions.
	if len(xt.SplitType) == 0 {
		if len(sets) > 0 {
			return nil, errors.New(
				"model has categorical data (categories_nodes) but no split_type",
			)
		}
		return sets, nil
	}

	if int64(len(xt.SplitType)) != numNodes {
		return nil, fmt.Errorf(
			"split_type length %d does not match num_nodes %d",
			len(xt.SplitType),
			numNodes,
		)
	}

	// Every node that split_type marks as categorical must have a decoded set,
	// and vice versa; a mismatch means we would treat a node as the wrong split
	// kind. Any split_type other than 0 (numeric) or 1 (categorical) is an
	// encoding we do not understand, so reject it rather than defaulting it to
	// numeric.
	for i := range numNodes {
		switch xt.SplitType[i] {
		case 0, 1:
		default:
			return nil, fmt.Errorf(
				"node %d has unsupported split_type %d",
				i,
				xt.SplitType[i],
			)
		}
		_, hasSet := sets[int(i)]
		isCategorical := xt.SplitType[i] == 1
		if hasSet != isCategorical {
			return nil, fmt.Errorf(
				"node %d has split_type %d but %s in categories_nodes",
				i,
				xt.SplitType[i],
				presence(hasSet),
			)
		}
	}

	return sets, nil
}

func presence(present bool) string {
	if present {
		return "is present"
	}
	return "is absent"
}

func parseModel(
	file string,
) (*XGBModel, []*Tree, error) {
	buf, err := os.ReadFile(filepath.Clean(file))
	if err != nil {
		return nil, nil, fmt.Errorf("reading file: %w", err)
	}

	var xm XGBModel
	if err := json.Unmarshal(sanitizeNonFiniteNumbers(buf), &xm); err != nil {
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

	categories, err := categorySets(xt, numNodes)
	if err != nil {
		return nil, err
	}

	nodes := make([]Node, numNodes)
	for i := range numNodes {
		cats, categorical := categories[int(i)]
		sc := float32(xt.SplitConditions[i])

		left := xt.LeftChildren[i]
		isLeaf := left == -1

		if err := checkSplitCondition(int(i), sc, categorical, isLeaf); err != nil {
			return nil, err
		}

		nodes[i].Data = NodeData{
			BaseWeight:     xt.BaseWeights[i],
			DefaultLeft:    xt.DefaultLeft[i] == 1,
			ID:             int(i),
			SplitCondition: sc,
			SplitIndex:     xt.SplitIndices[i],
			SumHessian:     xt.SumHessian[i],
			Categorical:    categorical,
			Categories:     cats,
		}

		right := xt.RightChildren[i]

		if isLeaf {
			continue
		}

		nodes[i].Left = &nodes[left]
		nodes[i].Right = &nodes[right]
	}

	return &Tree{
		Nodes:    nodes,
		NumNodes: int(numNodes),
	}, nil
}
