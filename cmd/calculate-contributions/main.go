// This program exercises the port of XGB feature contributions calculations.
package main

import (
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"github.com/maxmind/xgbshap"
)

func main() {
	modelFile := flag.String(
		"model",
		"",
		"Path to the XGBoost model JSON file",
	)

	featuresFile := flag.String(
		"features",
		"",
		"Path to a file containing features (CSV, one row being one set of features)",
	)

	ntreeLimit := flag.Int(
		"ntree-limit",
		0,
		"ntree limit to use. Newer model files have this included, so it is optional for those. Others are set in meta.json. Provide it here if necessary or to override",
	)

	flag.Parse()

	if *modelFile == "" || *featuresFile == "" {
		flag.Usage()
		os.Exit(1)
	}

	featuresSets, contributionsSets, err := predictContributionsBatch(
		*modelFile,
		*featuresFile,
		*ntreeLimit,
	)
	if err != nil {
		log.Fatal(err)
	}

	for i, contributionsSet := range contributionsSets {
		fmt.Printf("Feature set %d:\n", i)
		for j, feature := range featuresSets[i] {
			if feature == nil {
				fmt.Printf("  Feature %d: missing\n", j)
			} else {
				fmt.Printf("  Feature %d: %.6f\n", j, *feature)
			}
		}
		fmt.Printf("Contributions for feature set %d:\n", i)
		for j, contribution := range contributionsSet {
			fmt.Printf("  Contribution %d: %.6f\n", j, contribution)
		}
	}
}

func predictContributionsBatch(
	modelFile,
	featuresFile string,
	ntreeLimit int,
) ([][]*float32, [][]float32, error) {
	predictor, err := xgbshap.NewPredictor(
		modelFile,
		xgbshap.NtreeLimit(ntreeLimit),
	)
	if err != nil {
		return nil, nil, err
	}

	featuresSets, err := loadFeatures(featuresFile)
	if err != nil {
		return nil, nil, err
	}

	var contributionsSets [][]float32
	for _, features := range featuresSets {
		contribs, err := predictor.PredictContributions(features)
		if err != nil {
			return nil, nil, err
		}

		contributionsSets = append(
			contributionsSets,
			contribs[:len(contribs)-1], // Last value is not a contribution.
		)
	}

	return featuresSets, contributionsSets, nil
}

func loadFeatures(filename string) ([][]*float32, error) {
	fh, err := os.Open(filepath.Clean(filename))
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer fh.Close()

	csvReader := csv.NewReader(fh)
	var featuresSets [][]*float32
	for {
		record, err := csvReader.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("reading: %w", err)
		}

		var features []*float32
		for _, col := range record {
			if col == "" {
				features = append(features, nil) // Indicates a missing value.
				continue
			}

			feature, err := strconv.ParseFloat(col, 32)
			if err != nil {
				return nil, fmt.Errorf("parsing float: %w", err)
			}
			feature32 := float32(feature)

			features = append(features, &feature32)
		}

		featuresSets = append(featuresSets, features)
	}

	return featuresSets, nil
}
