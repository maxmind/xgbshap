package xgbshap

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func BenchmarkParseModel(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _, err := parseModel("testdata/small-model/model.json")
		require.NoError(b, err)
	}
}
