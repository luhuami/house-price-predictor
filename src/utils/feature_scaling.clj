(ns utils.feature-scaling
  (:require [clojure.core.matrix :as matrix]))

(defn- scale-column [matrix column]
  (let [mean (/ (matrix/esum column) (matrix/ecount column))
        range (- (matrix/emax column) (matrix/emin column))
        scaled-column (matrix/emap #(/ (- % mean) range) column)]
    (matrix/join-along 1 matrix scaled-column)))

(defn- scale-matrix [X]
  (reduce
    scale-column
    (matrix/broadcast [nil] [(matrix/row-count X) 1])
    (matrix/columns X)))

(defn scale [X]
  (let [scaled-matrix (scale-matrix X)
        last-column-index (dec (matrix/column-count scaled-matrix))]
    (matrix/matrix (matrix/submatrix scaled-matrix 1 [1 last-column-index]))))