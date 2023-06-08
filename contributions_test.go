package xgbshap

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPredictContributions(t *testing.T) {
	p, err := NewPredictor("testdata/small-model/model.json")
	require.NoError(t, err)

	features := []*float32{
		toPtr(2.109e+01),
		toPtr(2.657e+01),
		toPtr(1.427e+02),
		toPtr(1.311e+03),
		toPtr(1.141e-01),
		nil,
		toPtr(2.487e-01),
		toPtr(1.496e-01),
		nil,
		toPtr(7.398e-02),
		toPtr(6.298e-01),
		toPtr(7.629e-01),
		toPtr(4.414e+00),
		toPtr(8.146e+01),
		toPtr(4.253e-03),
		toPtr(4.759e-02),
		toPtr(3.872e-02),
		toPtr(1.567e-02),
		nil,
		toPtr(5.295e-03),
		toPtr(2.668e+01),
		toPtr(3.348e+01),
		toPtr(1.765e+02),
		toPtr(2.089e+03),
		nil,
		toPtr(7.584e-01),
		toPtr(6.780e-01),
		nil,
		toPtr(4.098e-01),
		toPtr(1.284e-01),
	}

	contributions, err := p.PredictContributions(features)
	require.NoError(t, err)

	assert.Equal(
		t,
		[]float32{
			-0.4111597,
			-0.16064933,
			-0.10974462,
			-0.31500122,
			-0.12911752,
			0.043756098,
			-0.043589786,
			-0.76819974,
			0.00020755455,
			0.007354198,
			-0.16104992,
			-0.044893354,
			-0.05673621,
			-0.20475619,
			0.10678359,
			-0.027902542,
			0,
			0.0064359917,
			-0.013715139,
			0.0105918255,
			-0.6475189,
			-0.37390453,
			-1.1507502,
			-0.7739394,
			-0.12558897,
			-0.13442251,
			-0.4490081,
			-0.41716415,
			-0.035092235,
			-0.091834456,
		},
		// Last element is bias, not a contribution.
		contributions[:len(contributions)-1],
	)
}

func toPtr(f float32) *float32 { return &f }
