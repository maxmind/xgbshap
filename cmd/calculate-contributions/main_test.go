package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPredictContributions(t *testing.T) {
	_, contributionsSets, err := predictContributionsBatch(
		"../../testdata/small-model/model.json",
		"../../testdata/small-model/features.csv",
		0,
	)
	require.NoError(t, err)

	require.Len(t, contributionsSets, 2)

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
		contributionsSets[0],
	)
}
