package xgbshap

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
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

func TestPredictContributionsCategorical(t *testing.T) {
	// This model has a single tree whose root is a categorical split on
	// feature 0 with categories {1, 3} routing right and default_left=1.
	// Left leaf base_weight=10, right leaf base_weight=30, equal hessians
	// (1, 1), so the mean value is 20.
	p, err := NewPredictor("testdata/categorical/model.json")
	require.NoError(t, err)

	t.Run("in-set value routes right", func(t *testing.T) {
		contributions, err := p.PredictContributions([]*float32{toPtr(1.0)})
		require.NoError(t, err)

		// contribution=10, bias=20; sum = 30 = right leaf.
		assert.Equal(t, []float32{10.0}, contributions[:len(contributions)-1])
	})

	t.Run("not-in-set value routes left", func(t *testing.T) {
		contributions, err := p.PredictContributions([]*float32{toPtr(2.0)})
		require.NoError(t, err)

		// contribution=-10, bias=20; sum = 10 = left leaf.
		assert.Equal(t, []float32{-10.0}, contributions[:len(contributions)-1])
	})

	t.Run("missing feature routes left via default_left", func(t *testing.T) {
		contributions, err := p.PredictContributions([]*float32{nil})
		require.NoError(t, err)

		// contribution=-10, bias=20; sum = 10 = left leaf.
		assert.Equal(t, []float32{-10.0}, contributions[:len(contributions)-1])
	})
}

func TestPredictContributionsNegInfSplit(t *testing.T) {
	// This model has a single tree whose root splits on feature 0 at -Infinity
	// with default_left=1. That means missing values route left (base_weight=10)
	// and all present values route right (base_weight=20). With equal hessians
	// (1, 1) the mean value is 15.
	p, err := NewPredictor("testdata/neg-inf-split/model.json")
	require.NoError(t, err)

	t.Run("present feature routes right", func(t *testing.T) {
		contributions, err := p.PredictContributions([]*float32{toPtr(5.0)})
		require.NoError(t, err)

		// contribution=5, bias=15; sum = 20 = right leaf.
		assert.Equal(t, []float32{5.0}, contributions[:len(contributions)-1])
	})

	t.Run("missing feature routes left", func(t *testing.T) {
		contributions, err := p.PredictContributions([]*float32{nil})
		require.NoError(t, err)

		// contribution=-5, bias=15; sum = 10 = left leaf.
		assert.Equal(t, []float32{-5.0}, contributions[:len(contributions)-1])
	})
}

func TestPredictContributionsRoundtrip(t *testing.T) {
	p, err := NewPredictor("testdata/roundtrip/model.json")
	require.NoError(t, err)

	allFeatures, err := readFeaturesCSV("testdata/roundtrip/features.csv")
	require.NoError(t, err)

	allContribs, err := readContributionsCSV("testdata/roundtrip/contributions.csv")
	require.NoError(t, err)

	require.Len(
		t,
		allContribs,
		len(allFeatures),
		"feature and contribution row counts must match",
	)

	const tolerance = 1e-5

	for row, features := range allFeatures {
		got, err := p.PredictContributions(features)
		require.NoError(t, err)

		expected := allContribs[row]

		// The last element is the bias. xgbshap does not yet add base_score
		// to it, so it will differ from XGBoost's output. Skip it.
		nFeatures := len(features)
		require.Len(t, got, nFeatures+1, "row %d: unexpected contribution length", row)
		require.Len(t, expected, nFeatures+1, "row %d: unexpected golden length", row)

		for col := range nFeatures {
			absDelta := math.Abs(float64(got[col] - expected[col]))
			relDelta := absDelta / math.Max(math.Abs(float64(expected[col])), 1.0)

			if relDelta > tolerance && absDelta > tolerance {
				t.Errorf(
					"row %d, feature %d: got %g, want %g (absDelta=%g, relDelta=%g)",
					row, col, got[col], expected[col], absDelta, relDelta,
				)
			}
		}
	}
}

func readFeaturesCSV(path string) ([][]*float32, error) {
	f, err := os.Open(path) //nolint:gosec // path is a test fixture, not user input
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	rows := make([][]*float32, len(records))
	for i, record := range records {
		row := make([]*float32, len(record))
		for j, cell := range record {
			if cell == "" {
				continue // nil means missing
			}
			v, err := strconv.ParseFloat(cell, 32)
			if err != nil {
				return nil, fmt.Errorf("row %d col %d: %w", i, j, err)
			}
			f32 := float32(v)
			row[j] = &f32
		}
		rows[i] = row
	}

	return rows, nil
}

func readContributionsCSV(path string) ([][]float32, error) {
	f, err := os.Open(path) //nolint:gosec // path is a test fixture, not user input
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	rows := make([][]float32, len(records))
	for i, record := range records {
		row := make([]float32, len(record))
		for j, cell := range record {
			v, err := strconv.ParseFloat(cell, 32)
			if err != nil {
				return nil, fmt.Errorf("row %d col %d: %w", i, j, err)
			}
			row[j] = float32(v)
		}
		rows[i] = row
	}

	return rows, nil
}

func toPtr(f float32) *float32 { return &f }
