package xgbshap

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func BenchmarkParseModel(b *testing.B) {
	for b.Loop() {
		_, _, err := parseModel("testdata/small-model/model.json")
		require.NoError(b, err)
	}
}
