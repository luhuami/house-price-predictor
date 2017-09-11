(ns feature-scaling
  (:require [clojure.core.matrix :as matrix]))

(defn- scale-column [matrix column]
  (let [mean (/ (matrix/esum column) (matrix/ecount column))
        max (matrix/emax column)
        scaled-column (matrix/emap #(/ (- % mean) max) column)]
    (matrix/join-along 1 matrix scaled-column)))

;create an empty matrix (nil) to combine scaled columns
(defn scale [X]
  (reduce
    scale-column
    (matrix/broadcast [nil] [(matrix/row-count X) 1])
    (matrix/columns X)))
