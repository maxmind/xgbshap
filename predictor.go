// Package xgbshap calculates feature contributions for XGBoost models.
package xgbshap

// Much of this code is ported from the xgboost C++ code.
//
// Copyright by XGBoost Contributors 2017-2023
//
// xgboost's code is Apache 2.0 licensed.

import (
	"fmt"
)

// Options holds Predictor options.
type Options struct {
	ntreeLimit int
}

// Option is a configuration function.
type Option func(*Options)

// NtreeLimit sets the ntree limit.
//
// For newer XGBoost models, this is found in the model file, so it does not
// need to be provided.
func NtreeLimit(ntreeLimit int) func(*Options) {
	return func(o *Options) {
		o.ntreeLimit = ntreeLimit
	}
}

// Predictor calculates feature contributions for an XGBoost model.
type Predictor struct {
	ntreeLimit int
	trees      []*Tree
}

// NewPredictor creates a Predictor.
func NewPredictor(
	modelFile string,
	opts ...Option,
) (*Predictor, error) {
	var o Options
	for _, f := range opts {
		f(&o)
	}

	xgbModel, trees, err := parseModel(modelFile)
	if err != nil {
		return nil, err
	}

	if o.ntreeLimit == 0 {
		o.ntreeLimit, err = resolveNtreeLimit(
			xgbModel.Learner.Attributes,
			len(trees),
		)
		if err != nil {
			return nil, err
		}
	}

	return &Predictor{
		ntreeLimit: o.ntreeLimit,
		trees:      trees,
	}, nil
}

// resolveNtreeLimit determines how many trees to use when the caller has not
// set an explicit limit. Older XGBoost versions store best_ntree_limit
// directly; newer ones store best_iteration (0-based), so the number of trees
// to use is best_iteration + 1. When neither attribute is present there was no
// early stopping, so all trees are used. best_ntree_limit takes precedence when
// both are present.
func resolveNtreeLimit(attrs Attributes, numTrees int) (int, error) {
	switch {
	case attrs.BestNtreeLimit != "":
		n, err := attrs.BestNtreeLimit.Int64()
		if err != nil {
			return 0, fmt.Errorf("parsing best_ntree_limit: %w", err)
		}
		return int(n), nil
	case attrs.BestIteration != "":
		n, err := attrs.BestIteration.Int64()
		if err != nil {
			return 0, fmt.Errorf("parsing best_iteration: %w", err)
		}
		return int(n) + 1, nil
	default:
		return numTrees, nil
	}
}
