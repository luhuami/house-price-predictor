(ns logistic-regression
  (:require [clojure.math.numeric-tower :as math]))

(defn sigmoid [z]
  (->> z
       (math/expt Math/E)
       (/ 1)
       (+ 1)
       (/ 1)))
