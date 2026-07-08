package xgbshap

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSanitizeNonFiniteNumbers(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "no tokens is unchanged",
			in:   `{"split_conditions":[1.5,-2.0]}`,
			want: `{"split_conditions":[1.5,-2.0]}`,
		},
		{
			name: "quotes the non-finite tokens",
			in:   `[1.5, -Infinity, Infinity, NaN, 2.0]`,
			want: `[1.5, "-Infinity", "Infinity", "NaN", 2.0]`,
		},
		{
			// -Infinity must be matched as a whole so the minus sign is not left
			// behind as a separate token.
			name: "negative infinity keeps its sign inside the quotes",
			in:   `[-Infinity]`,
			want: `["-Infinity"]`,
		},
		{
			// A feature literally named with a token must not be rewritten; it is
			// inside a JSON string, not a numeric value.
			name: "tokens inside strings are left alone",
			in:   `{"feature_names":["NaN","Infinity"],"split_conditions":[NaN]}`,
			want: `{"feature_names":["NaN","Infinity"],"split_conditions":["NaN"]}`,
		},
		{
			name: "escaped quote does not end the string early",
			in:   `{"name":"a\"NaN","v":[NaN]}`,
			want: `{"name":"a\"NaN","v":["NaN"]}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(
				t,
				test.want,
				string(sanitizeNonFiniteNumbers([]byte(test.in))),
			)
		})
	}
}

func TestXGBFloatUnmarshalJSON(t *testing.T) {
	// After sanitizeNonFiniteNumbers runs, split_conditions is a mix of JSON
	// numbers and quoted non-finite tokens; both forms must decode.
	var got []xgbFloat
	err := json.Unmarshal(
		[]byte(`[1.5, "-Infinity", "Infinity", "NaN", -2.0]`),
		&got,
	)
	require.NoError(t, err)

	require.Len(t, got, 5)
	assert.InDelta(t, 1.5, float64(got[0]), 1e-6)
	assert.True(t, math.IsInf(float64(got[1]), -1))
	assert.True(t, math.IsInf(float64(got[2]), 1))
	assert.True(t, math.IsNaN(float64(got[3])))
	assert.InDelta(t, -2.0, float64(got[4]), 1e-6)

	t.Run("quoted non-numeric string is rejected", func(t *testing.T) {
		err := json.Unmarshal([]byte(`["bogus"]`), &[]xgbFloat{})
		require.Error(t, err)
	})
	t.Run("non-numeric bare token is rejected", func(t *testing.T) {
		err := json.Unmarshal([]byte(`[true]`), &[]xgbFloat{})
		require.Error(t, err)
	})
}

func TestParseModelNegInfSplit(t *testing.T) {
	_, trees, err := parseModel("testdata/neg-inf-split/model.json")
	require.NoError(t, err)

	require.Len(t, trees, 1)

	root := trees[0].Nodes[0]
	assert.True(
		t,
		math.IsInf(float64(root.Data.SplitCondition), -1),
		"root split condition should be -Infinity",
	)
	assert.True(t, root.Data.DefaultLeft)
	assert.Equal(t, float32(10.0), root.Left.Data.BaseWeight)
	assert.Equal(t, float32(20.0), root.Right.Data.BaseWeight)
}

func TestParseModelCategorical(t *testing.T) {
	_, trees, err := parseModel("testdata/categorical/model.json")
	require.NoError(t, err)

	require.Len(t, trees, 1)

	root := trees[0].Nodes[0]
	assert.True(t, root.Data.Categorical)
	assert.Equal(t, []int{1, 3}, root.Data.Categories)
	assert.True(t, root.Data.DefaultLeft)
	assert.Equal(t, float32(10.0), root.Left.Data.BaseWeight)
	assert.Equal(t, float32(30.0), root.Right.Data.BaseWeight)

	// Leaf nodes must not be marked categorical.
	assert.False(t, root.Left.Data.Categorical)
	assert.False(t, root.Right.Data.Categorical)
}

func TestCategorySets(t *testing.T) {
	// baseTree is a valid two-categorical-node tree the error cases mutate.
	baseTree := func() XGBTree {
		xt := XGBTree{
			SplitType:          []int{1, 1, 0},
			Categories:         []int{0, 2, 1, 3, 4},
			CategoriesNodes:    []int{0, 1},
			CategoriesSegments: []int{0, 2},
			CategoriesSizes:    []int{2, 3},
		}
		xt.TreeParam.NumNodes = "3"
		return xt
	}

	t.Run("valid decoding", func(t *testing.T) {
		sets, err := categorySets(baseTree(), 3)
		require.NoError(t, err)
		assert.Equal(
			t,
			map[int][]int{0: {0, 2}, 1: {1, 3, 4}},
			sets,
		)
	})

	t.Run("no categorical splits returns empty", func(t *testing.T) {
		xt := XGBTree{SplitType: []int{0, 0, 0}}
		xt.TreeParam.NumNodes = "3"
		sets, err := categorySets(xt, 3)
		require.NoError(t, err)
		assert.Empty(t, sets)
	})

	tests := []struct {
		name   string
		mutate func(xt *XGBTree)
	}{
		{
			name: "mismatched segment length",
			mutate: func(xt *XGBTree) {
				xt.CategoriesSegments = []int{0}
			},
		},
		{
			name: "mismatched size length",
			mutate: func(xt *XGBTree) {
				xt.CategoriesSizes = []int{2}
			},
		},
		{
			name: "segment out of range",
			mutate: func(xt *XGBTree) {
				xt.CategoriesSizes = []int{2, 99}
			},
		},
		{
			name: "negative segment start",
			mutate: func(xt *XGBTree) {
				xt.CategoriesSegments = []int{-1, 2}
			},
		},
		{
			name: "split_type length mismatch",
			mutate: func(xt *XGBTree) {
				xt.SplitType = []int{1, 1}
			},
		},
		{
			name: "split_type categorical but no set",
			mutate: func(xt *XGBTree) {
				xt.SplitType = []int{1, 1, 1}
			},
		},
		{
			name: "set present but split_type numeric",
			mutate: func(xt *XGBTree) {
				xt.SplitType = []int{0, 1, 0}
			},
		},
		{
			name: "node ID out of range",
			mutate: func(xt *XGBTree) {
				xt.CategoriesNodes = []int{0, 99}
			},
		},
		{
			name: "negative node ID",
			mutate: func(xt *XGBTree) {
				xt.CategoriesNodes = []int{0, -1}
			},
		},
		{
			name: "categorical data but no split_type",
			mutate: func(xt *XGBTree) {
				xt.SplitType = nil
			},
		},
		{
			name: "unsupported split_type value",
			mutate: func(xt *XGBTree) {
				xt.SplitType = []int{2, 1, 0}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			xt := baseTree()
			test.mutate(&xt)
			_, err := categorySets(xt, 3)
			require.Error(t, err)
		})
	}
}

func TestCheckSplitCondition(t *testing.T) {
	t.Run("finite threshold accepted", func(t *testing.T) {
		require.NoError(t, checkSplitCondition(0, 1.5, false, false))
	})
	t.Run("negative infinity accepted", func(t *testing.T) {
		require.NoError(t, checkSplitCondition(0, float32(math.Inf(-1)), false, false))
	})
	t.Run("positive infinity rejected", func(t *testing.T) {
		require.Error(t, checkSplitCondition(0, float32(math.Inf(1)), false, false))
	})
	t.Run("NaN rejected", func(t *testing.T) {
		require.Error(t, checkSplitCondition(0, float32(math.NaN()), false, false))
	})
	t.Run("leaf nodes skip validation", func(t *testing.T) {
		require.NoError(t, checkSplitCondition(0, float32(math.NaN()), false, true))
		require.NoError(t, checkSplitCondition(0, float32(math.Inf(1)), false, true))
	})
	t.Run("categorical nodes skip threshold validation", func(t *testing.T) {
		require.NoError(t, checkSplitCondition(0, float32(math.NaN()), true, false))
		require.NoError(t, checkSplitCondition(0, float32(math.Inf(1)), true, false))
	})
}

func BenchmarkParseModel(b *testing.B) {
	for b.Loop() {
		_, _, err := parseModel("testdata/small-model/model.json")
		require.NoError(b, err)
	}
}
