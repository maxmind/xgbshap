# xgb2shap

This is a Go package for calculating feature contributions for
[XGBoost](https://github.com/dmlc/xgboost) models. The code is ported from
the XGBoost C++ code. This package exists to make it possible to calculate
contributions with pure Go code rather than having to use cgo or another
language.

## Missing Functionality

While the code is ported from the XGBoost C++ code, not all code was
ported. If you find results differ between the two implementations, it is
possible that code relevant to your model was not ported.

In particular, we have not yet ported:

* Code involving condition and condition_features
* Code involving tree weights
* Some of the code involving the bias
* Code involving categories

It is also possible that XGBoost's code has changed since this code was
written. We will be attempting to keep this implementation up to date.

## Example Usage

```go
modelFile := "/path/to/model.json"

predictor, err := xgbshap.NewPredictor(modelFile)
if err != nil {
    return err
}

features := []*float32{...your features here...}

contributions, err := predictor.PredictContributions(features)
if err != nil {
    return err
}
```

## Bug Reports

Please report bugs by filing an issue with our GitHub issue tracker at
[https://github.com/maxmind/xgbshap/issues](https://github.com/maxmind/xgbshap/issues).

## Copyright and License

This software is Copyright (c) 2023 by MaxMind, Inc.

Much of the code is ported from the XGBoost project, so it is also
Copyright (c) 2017-2023 by XGBoost Contributors.

This is free software, licensed under the [Apache License, Version
2.0](LICENSE).
