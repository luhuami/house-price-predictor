(ns utils.feature-normalize
  (:require [clojure.core.matrix :as matrix]))

(defn- calc-column-mean [column]
  (/ (matrix/esum column) (matrix/ecount column)))

(defn- calc-column-range [column]
  (- (matrix/emax column) (matrix/emin column)))

(defn- normalize-column [column]
  (let [mean (calc-column-mean column)
        range (calc-column-range column)]
    (matrix/emap #(/ (- % mean) range) column)))

(defn- create-normalize-params [column]
  (matrix/matrix [[(calc-column-mean column)] [(calc-column-range column)]]))

(defn- map-columns [f]
  (fn [X]
    (matrix/transpose (map f (matrix/columns X)))))

(defn normalize [X]
  ((map-columns normalize-column) X))

(defn calc-normalize-params [X]
  ((map-columns create-normalize-params) X))