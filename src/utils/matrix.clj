(ns utils.matrix
  (:require [clojure.core.matrix :as matrix]))

(defn create-matrix-with-value [matrix-shape-vec v]
  (matrix/fill (matrix/new-matrix (first matrix-shape-vec) (last matrix-shape-vec)) v))

(defn remove-first-column [m]
  (if (> (matrix/column-count m) 1)
    (matrix/matrix
      (matrix/submatrix m 1 [1 (dec (matrix/column-count m))]))))