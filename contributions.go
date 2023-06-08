package xgbshap

// Much of this code is ported from the xgboost C++ code.
//
// # Copyright by XGBoost Contributors 2017-2023
//
// xgboost's code is Apache 2.0 licensed.

import (
	"fmt"
)

// PredictContributions calculates the contributions of features.
func (p *Predictor) PredictContributions(
	features []*float32,
) ([]float32, error) {
	return predictContributions(
		p.ntreeLimit,
		p.trees,
		features,
	)
}

// Calculate the contributions of features.
//
// This is a port of the xgboost code. Specifically, the flow we're porting is
// this:
//   - In MM-XGBoost, our Perl XS package, we call mm_xg_model_predict(). This
//     creates a dmatrix (data matrix), which holds our one dimensional array
//     of features. Here we use a slice of *float32 for that purpose.
//   - This calls XGBoosterPredict() in xgboost's c_api.cc. We call it with
//     option_mask=4 as we want contributions.
//   - This calls learner->Predict() in learner.cc.
//   - This calls gbm_->PredictContribution() in cpu_predictor.cc.
//   - This calls CalculateContributions() in cpu_treeshap.cc, which is the main
//     algorithm for calculating contributions.
//
// This function is equivalent to PredictContribution() in xgboost.
func predictContributions(
	ntreeLimit int,
	trees []*Tree,
	features []*float32,
) ([]float32, error) {
	// The main entrypoint is the call to Predict():
	//
	// In the C++ code, iterationEnd gets set by a function. However that
	// function seems to be a no-op for our models and the value will be
	// nTreeLimit.
	//
	// learner->Predict(*static_cast<std::shared_ptr<DMatrix> *>(dmat),
	//                 (option_mask & 1) != 0, &entry.predictions, 0, iteration_end,
	//                 static_cast<bool>(training), (option_mask & 2) != 0,
	//                 (option_mask & 4) != 0, (option_mask & 8) != 0,
	//                 (option_mask & 16) != 0);

	// Predict() calls PredictContribution():
	//
	// Its parameters:
	// - data.get is our input features (data matrix)
	// - out_preds is an array of floats (where we store the output)
	// - layer_begin = 0
	// - layer_end = iterationEnd
	// - approx_contrib is a bool and it's always false in our calls
	// gbm_->PredictContribution(data.get(), out_preds, layer_begin, layer_end, approx_contribs);

	// ngroup seems to always be 1.

	// allocate space for (number of features + bias) times the number of rows
	//
	// +1 for "bias" (xgboost's term in its source) or "intercept term" (what we
	// refer to it in our Perl code). It's an extra value we get when calculating
	// contributions, which we don't use.
	nColumns := len(features) + 1

	contribs := make([]float32, nColumns)

	// Initialize tree node mean values.
	meanValues := make([][]float32, ntreeLimit)
	for i := 0; i < ntreeLimit; i++ {
		meanValues[i] = make([]float32, trees[i].NumNodes)

		nodeIndex := 0
		fillNodeMeanValues(trees[i], nodeIndex, meanValues[i])
	}

	// base_score/base_margin seem to only be used for calculating the
	// bias/intercept, so I'm ignoring them for now as we don't use those as far
	// as I know.

	// The C++ code processes the features in batches (the GetBatches() loop). In
	// the case where we're calculating contributions for one feature set,
	// there's only one batch, so we don't need to worry about that.

	// If ngroup was not 1, then we'd need an additional loop here.

	for i := 0; i < ntreeLimit; i++ {
		treeMeanValues := meanValues[i]

		treeContribs := make([]float32, nColumns)

		// I'm not sure what condition and condition_feature parameters are. They
		// are 0 in my testing.
		var condition, conditionFeature int

		err := calculateContributions(
			trees[i],
			features,
			treeMeanValues,
			treeContribs,
			condition,
			conditionFeature,
		)
		if err != nil {
			return nil, err
		}

		for ci := 0; ci < nColumns; ci++ {
			// tree_weights is null in my testing, so I'm ignoring it.
			contribs[ci] += treeContribs[ci]
		}

		// As mentioned above, since we don't use bias/intercept, I omit the code
		// for that.
	}

	return contribs, nil
}

// This is equivalent to the two FillNodeMeanValues() functions in xgboost.
func fillNodeMeanValues(
	tree *Tree,
	nodeIndex int,
	meanValues []float32,
) float32 {
	node := tree.Nodes[nodeIndex]

	var result float32
	if node.IsLeaf() {
		result = node.LeafValue()
	} else {
		result = fillNodeMeanValues(
			tree,
			node.Left.Data.ID,
			meanValues,
		) * node.Left.Data.SumHessian

		result += fillNodeMeanValues(
			tree,
			node.Right.Data.ID,
			meanValues,
		) * node.Right.Data.SumHessian

		result /= node.Data.SumHessian
	}

	meanValues[nodeIndex] = result

	return result
}

// PathElement is an element used by the treeshap algorithm.
type PathElement struct {
	FeatureIndex int
	ZeroFraction float32
	OneFraction  float32
	Pweight      float32
}

// This is equivalent to CalculateContributions() in xgboost.
func calculateContributions(
	tree *Tree,
	features []*float32,
	meanValues,
	contribs []float32,
	condition,
	conditionFeature int,
) error {
	// Find the expected value of the tree's predictions
	if condition == 0 {
		nodeValue := meanValues[0]
		contribs[len(features)] += nodeValue
	}

	// Preallocate space for the unique path data
	//
	// I'm not sure what the +2 is for.
	maxDepth := tree.Nodes[0].MaxDepth() + 2
	uniquePathData := make([]PathElement, (maxDepth*(maxDepth+1))/2)

	var nodeIndex, uniqueDepth int
	parentZeroFraction := float32(1)
	parentOneFraction := float32(1)
	parentFeatureIndex := -1
	conditionFraction := float32(1)

	return treeShap(
		tree,
		features,
		contribs,
		nodeIndex,
		uniqueDepth,
		uniquePathData,
		parentZeroFraction,
		parentOneFraction,
		parentFeatureIndex,
		condition,
		conditionFeature,
		conditionFraction,
	)
}

// Recursive function that computes the feature attributions for a single tree.
//
// This is equivalent to TreeShap() in xgboost.
func treeShap(
	tree *Tree,
	features []*float32,
	phi []float32, // AKA contribs
	nodeIndex,
	uniqueDepth int,
	parentUniquePath []PathElement,
	parentZeroFraction,
	parentOneFraction float32,
	parentFeatureIndex,
	condition,
	conditionFeature int,
	conditionFraction float32,
) error {
	node := tree.Nodes[nodeIndex]

	// stop if we have no weight coming down to us
	if conditionFraction == 0 {
		return nil
	}

	// extend the unique path
	uniquePath := parentUniquePath[uniqueDepth+1:]
	copy(uniquePath, parentUniquePath[:uniqueDepth+1])

	if condition == 0 || conditionFeature != parentFeatureIndex {
		extendPath(
			uniquePath,
			uniqueDepth,
			parentZeroFraction,
			parentOneFraction,
			parentFeatureIndex,
		)
	}

	splitIndex := node.Data.SplitIndex

	if node.IsLeaf() {
		for i := 1; i <= uniqueDepth; i++ {
			w, err := unwoundPathSum(uniquePath, uniqueDepth, i)
			if err != nil {
				return err
			}

			el := uniquePath[i]

			phi[el.FeatureIndex] += w *
				(el.OneFraction - el.ZeroFraction) *
				node.LeafValue() *
				conditionFraction
		}

		return nil
	}

	// Internal node

	// find which branch is "hot" (meaning x would follow it)

	// The GetCategoriesMatrix call is apparently not used, at least in the model
	// I'm testing with. I think it is related to the categories field in the
	// JSON model. We always hit the false branch for the code involving it in
	// GetNextNode(). I'm omitting it for now.

	hasMissing := true                       // We always can have missing values.
	isMissing := features[splitIndex] == nil // nil means missing.
	hotIndex := getNextNode(
		hasMissing,
		node,
		nodeIndex,
		features[splitIndex],
		isMissing,
	)

	var coldIndex int
	if hotIndex == node.Left.Data.ID {
		coldIndex = node.Right.Data.ID
	} else {
		coldIndex = node.Left.Data.ID
	}

	w := node.Data.SumHessian
	hotZeroFraction := tree.Nodes[hotIndex].Data.SumHessian / w
	coldZeroFraction := tree.Nodes[coldIndex].Data.SumHessian / w

	incomingZeroFraction := float32(1)
	incomingOneFraction := float32(1)

	// see if we have already split on this feature,
	// if so we undo that split so we can redo it for this node
	var pathIndex int
	for ; pathIndex <= uniqueDepth; pathIndex++ {
		if uniquePath[pathIndex].FeatureIndex == splitIndex {
			break
		}
	}

	if pathIndex != uniqueDepth+1 {
		incomingZeroFraction = uniquePath[pathIndex].ZeroFraction
		incomingOneFraction = uniquePath[pathIndex].OneFraction
		unwindPath(uniquePath, uniqueDepth, pathIndex)
		uniqueDepth--
	}

	// divide up the conditionFraction among the recursive calls
	hotConditionFraction := conditionFraction
	coldConditionFraction := conditionFraction
	if condition > 0 && splitIndex == conditionFeature {
		coldConditionFraction = 0
		uniqueDepth--
	} else if condition < 0 && splitIndex == conditionFeature {
		hotConditionFraction *= hotZeroFraction
		coldConditionFraction *= coldZeroFraction
		uniqueDepth--
	}

	err := treeShap(
		tree,
		features,
		phi,
		hotIndex,
		uniqueDepth+1,
		uniquePath,
		hotZeroFraction*incomingZeroFraction,
		incomingOneFraction,
		splitIndex,
		condition,
		conditionFeature,
		hotConditionFraction,
	)
	if err != nil {
		return err
	}

	err = treeShap(
		tree,
		features,
		phi,
		coldIndex,
		uniqueDepth+1,
		uniquePath,
		coldZeroFraction*incomingZeroFraction,
		0,
		splitIndex,
		condition,
		conditionFeature,
		coldConditionFraction,
	)
	if err != nil {
		return err
	}

	return nil
}

// extend our decision path with a fraction of one and zero extensions
//
// This is equivalent to ExtendPath() in xgboost.
func extendPath(
	uniquePath []PathElement,
	uniqueDepth int,
	zeroFraction,
	oneFraction float32,
	featureIndex int,
) {
	uniquePath[uniqueDepth].FeatureIndex = featureIndex
	uniquePath[uniqueDepth].ZeroFraction = zeroFraction
	uniquePath[uniqueDepth].OneFraction = oneFraction

	if uniqueDepth == 0 {
		uniquePath[uniqueDepth].Pweight = 1
	} else {
		uniquePath[uniqueDepth].Pweight = 0
	}

	for i := uniqueDepth - 1; i >= 0; i-- {
		uniquePath[i+1].Pweight += oneFraction *
			uniquePath[i].Pweight *
			float32(i+1) /
			float32(uniqueDepth+1)

		uniquePath[i].Pweight = zeroFraction *
			uniquePath[i].Pweight *
			float32(uniqueDepth-i) /
			float32(uniqueDepth+1)
	}
}

// determine what the total permutation weight would be if
// we unwound a previous extension in the decision path
//
// This is equivalent to UnwoundPathSum() in xgboost.
func unwoundPathSum(
	uniquePath []PathElement,
	uniqueDepth,
	pathIndex int,
) (float32, error) {
	oneFraction := uniquePath[pathIndex].OneFraction
	zeroFraction := uniquePath[pathIndex].ZeroFraction
	nextOnePortion := uniquePath[uniqueDepth].Pweight

	var total float32
	for i := uniqueDepth - 1; i >= 0; i-- {
		if oneFraction != 0 {
			tmp := nextOnePortion *
				float32(uniqueDepth+1) /
				(float32(i+1) * oneFraction)

			total += tmp

			nextOnePortion = uniquePath[i].Pweight -
				tmp*zeroFraction*
					(float32(uniqueDepth-i)/float32(uniqueDepth+1))

			continue
		}

		if zeroFraction != 0 {
			total += (uniquePath[i].Pweight / zeroFraction) /
				(float32(uniqueDepth-i) / float32(uniqueDepth+1))
			continue
		}

		if uniquePath[i].Pweight != 0 {
			return 0, fmt.Errorf("unique path %d must have zero weight", i)
		}
	}

	return total, nil
}

// This is equivalent to GetNextNode() in xgboost (predict_fn.h).
func getNextNode(
	hasMissing bool,
	node *Node,
	_ int, // node index
	featureValue *float32,
	isMissing bool,
) int { // Return node index
	if hasMissing && isMissing {
		if node.Data.DefaultLeft {
			return node.Left.Data.ID
		}
		return node.Right.Data.ID
	}

	// As I mention above, we don't currently need the "cats" (categories)
	// parameter. From what I can tell, it is not set in the models we use, at
	// least what I am testing with.

	nextNodeIndex := node.Left.Data.ID
	if !(*featureValue < node.Data.SplitCondition) {
		nextNodeIndex++
	}

	return nextNodeIndex
}

// undo a previous extension of the decision path
//
// This is equivalent to UnwindPath() in xgboost.
func unwindPath(
	uniquePath []PathElement,
	uniqueDepth,
	pathIndex int,
) {
	oneFraction := uniquePath[pathIndex].OneFraction
	zeroFraction := uniquePath[pathIndex].ZeroFraction
	nextOnePortion := uniquePath[uniqueDepth].Pweight

	for i := uniqueDepth - 1; i >= 0; i-- {
		if oneFraction != 0 {
			tmp := uniquePath[i].Pweight

			uniquePath[i].Pweight = nextOnePortion *
				float32(uniqueDepth+1) / (float32(i+1) * oneFraction)

			nextOnePortion = tmp -
				uniquePath[i].Pweight*
					zeroFraction*
					float32(uniqueDepth-i)/float32(uniqueDepth+1)
		} else {
			uniquePath[i].Pweight = (uniquePath[i].Pweight * float32(uniqueDepth+1)) /
				(zeroFraction * float32(uniqueDepth-i))
		}
	}

	for i := pathIndex; i < uniqueDepth; i++ {
		uniquePath[i].FeatureIndex = uniquePath[i+1].FeatureIndex
		uniquePath[i].ZeroFraction = uniquePath[i+1].ZeroFraction
		uniquePath[i].OneFraction = uniquePath[i+1].OneFraction
	}
}
