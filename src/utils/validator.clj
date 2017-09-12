(ns utils.validator
  (:require [clojure.core.matrix :as matrix]))

(defn- not-empty? [X y]
  (and
    (> (matrix/row-count X) 0)
    (> (matrix/column-count X) 0)
    (> (matrix/row-count y) 0)))

(defn- same-row-count? [X y]
  (= (matrix/row-count X) (matrix/row-count y)))

;vector and matrix are different. y must be a matrix rather a vector.
(defn valid? [X y]
  (and
    (matrix/matrix? X)
    (matrix/matrix? y)
    (not-empty? X y)
    (= 1 (matrix/column-count y))
    (same-row-count? X y)))