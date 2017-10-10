(ns utils.matrix
  (:require [clojure.core.matrix :as matrix]))

(defn create-matrix-with-value [matrix-shape-vec v]
  (matrix/fill (matrix/new-matrix (first matrix-shape-vec) (last matrix-shape-vec)) v))
