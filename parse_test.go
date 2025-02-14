package xgbshap

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func BenchmarkParseModel(b *testing.B) {
	for range b.N {
		_, _, err := parseModel("testdata/small-model/model.json")
		require.NoError(b, err)
	}
}
