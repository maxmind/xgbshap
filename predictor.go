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
		ntreeLimit64, err := xgbModel.Learner.Attributes.BestNtreeLimit.Int64()
		if err != nil {
			return nil, fmt.Errorf("getting best ntree limit as int64: %w", err)
		}

		o.ntreeLimit = int(ntreeLimit64)
	}

	return &Predictor{
		ntreeLimit: o.ntreeLimit,
		trees:      trees,
	}, nil
}
