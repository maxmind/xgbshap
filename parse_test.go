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

func TestCheckSplitCondition(t *testing.T) {
	t.Run("finite threshold accepted", func(t *testing.T) {
		require.NoError(t, checkSplitCondition(0, 1.5, false))
	})
	t.Run("negative infinity accepted", func(t *testing.T) {
		require.NoError(t, checkSplitCondition(0, float32(math.Inf(-1)), false))
	})
	t.Run("positive infinity rejected", func(t *testing.T) {
		require.Error(t, checkSplitCondition(0, float32(math.Inf(1)), false))
	})
	t.Run("NaN rejected", func(t *testing.T) {
		require.Error(t, checkSplitCondition(0, float32(math.NaN()), false))
	})
	t.Run("leaf nodes skip validation", func(t *testing.T) {
		require.NoError(t, checkSplitCondition(0, float32(math.NaN()), true))
		require.NoError(t, checkSplitCondition(0, float32(math.Inf(1)), true))
	})
}

func BenchmarkParseModel(b *testing.B) {
	for b.Loop() {
		_, _, err := parseModel("testdata/small-model/model.json")
		require.NoError(b, err)
	}
}
