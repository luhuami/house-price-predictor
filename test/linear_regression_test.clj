(ns linear-regression-test
    (:require [clojure.test :refer :all]
              [clojure.string :refer :all]
              [linear-regression :as lr]
              [utils.feature-normalize :as fs]
              [utils.read :as r]))

;ex1 uses alpha 0.01 and 1500 iteration without normalization. thata is [-3.630291 1.166362]
(defn- test-single-feature []
  (let[X (r/parse-X "test/resource/linear/regression/single-feature.txt" r/double-parser)
       y (r/parse-y "test/resource/linear/regression/single-feature.txt" r/double-parser)]
    (lr/perform-batch-gradient-decent X y 0.01 1500)))

(defn- test-multiple-feature []
  (let[X (r/parse-X "test/resource/linear/regression/multi-feature.txt" r/int-parser)
       y (r/parse-y "test/resource/linear/regression/multi-feature.txt" r/int-parser)]
    (lr/perform-batch-gradient-decent (fs/normalize X) y 0.01 1500)))
