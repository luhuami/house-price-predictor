(ns linear-regression-test
    (:require [clojure.test :refer :all]
              [clojure.string :refer :all]
              [clojure.core.matrix :as matrix]
              [linear-regression :as lr]))


(def row-data (slurp "test/resource/data.txt"))

(def row-arr (split-lines row-data))

(def arr (map #(split % #",") row-arr))

(def arr1 (flatten arr))

(def arr2 (map #(Integer/parseInt %) arr1))

(def x-list (partition 2 3 arr2))

(def X (matrix/matrix x-list))

(def y-list (map last (partition 3 arr2)))

(def y (matrix/matrix (map vector y-list)))



