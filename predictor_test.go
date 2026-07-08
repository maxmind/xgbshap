package xgbshap

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveNtreeLimit(t *testing.T) {
	const numTrees = 24

	t.Run("best_ntree_limit is used directly", func(t *testing.T) {
		n, err := resolveNtreeLimit(Attributes{BestNtreeLimit: "10"}, numTrees)
		require.NoError(t, err)
		assert.Equal(t, 10, n)
	})

	t.Run("best_iteration is offset by one", func(t *testing.T) {
		// best_iteration is 0-based, so 13 means 14 trees.
		n, err := resolveNtreeLimit(Attributes{BestIteration: "13"}, numTrees)
		require.NoError(t, err)
		assert.Equal(t, 14, n)
	})

	t.Run("best_iteration of zero means one tree", func(t *testing.T) {
		// The boundary value: best_iteration=0 (first iteration was best) is a
		// legitimate XGBoost output and must resolve to 1, not 0.
		n, err := resolveNtreeLimit(Attributes{BestIteration: "0"}, numTrees)
		require.NoError(t, err)
		assert.Equal(t, 1, n)
	})

	t.Run("neither attribute uses all trees", func(t *testing.T) {
		n, err := resolveNtreeLimit(Attributes{}, numTrees)
		require.NoError(t, err)
		assert.Equal(t, numTrees, n)
	})

	t.Run("best_ntree_limit takes precedence over best_iteration", func(t *testing.T) {
		n, err := resolveNtreeLimit(
			Attributes{BestNtreeLimit: "10", BestIteration: "13"},
			numTrees,
		)
		require.NoError(t, err)
		assert.Equal(t, 10, n)
	})

	t.Run("non-numeric best_ntree_limit errors", func(t *testing.T) {
		_, err := resolveNtreeLimit(Attributes{BestNtreeLimit: "bogus"}, numTrees)
		require.ErrorContains(t, err, "best_ntree_limit")
	})

	t.Run("non-numeric best_iteration errors", func(t *testing.T) {
		_, err := resolveNtreeLimit(Attributes{BestIteration: "bogus"}, numTrees)
		require.ErrorContains(t, err, "best_iteration")
	})
}

func TestNewPredictorResolvesNtreeLimit(t *testing.T) {
	// The roundtrip model stores best_iteration=13, so the predictor should use
	// 14 trees. This pins the end-to-end wiring of resolveNtreeLimit into
	// NewPredictor, including the best_iteration+1 offset.
	p, err := NewPredictor("testdata/roundtrip/model.json")
	require.NoError(t, err)
	assert.Equal(t, 14, p.ntreeLimit)
}

func TestNewPredictorExplicitNtreeLimitWins(t *testing.T) {
	// An explicit non-zero NtreeLimit bypasses attribute-based resolution.
	p, err := NewPredictor("testdata/roundtrip/model.json", NtreeLimit(5))
	require.NoError(t, err)
	assert.Equal(t, 5, p.ntreeLimit)
}

func TestNewPredictorNtreeLimitZeroFallsThrough(t *testing.T) {
	// NtreeLimit(0) is indistinguishable from not setting the option: zero is
	// treated as "unset" and resolution falls through to the model attributes.
	// It does not mean "use zero trees".
	p, err := NewPredictor("testdata/roundtrip/model.json", NtreeLimit(0))
	require.NoError(t, err)
	assert.Equal(t, 14, p.ntreeLimit)
}
