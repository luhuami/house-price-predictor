(ns utils.read
  (:require [clojure.test :refer :all]
            [clojure.string :refer :all]
            [clojure.core.matrix :as matrix]
            [linear-regression :as lr]
            [utils.feature-normalize :as fs]))

(defn- parse-to-array [file-path]
  (let [file (slurp file-path)
        line-seq (split-lines file)]
    (map #(split % #",") line-seq)))

(defn- parse-to-x-matrix [file-path]
  (let[arr (parse-to-array file-path)
       column-num (dec (count (first arr)))]
    (matrix/submatrix (matrix/matrix arr) 1 [0 column-num])))

(defn- parse-to-y-matrix [file-path]
  (let[arr (parse-to-array file-path)
       column-num (dec (count (first arr)))]
    (matrix/submatrix (matrix/matrix arr) 1 [column-num 1])))

(defn parse-y [file-path type-func]
  (let [y (parse-to-y-matrix file-path)]
    (matrix/matrix (matrix/emap type-func y))))

(defn parse-X [file-path type-func]
  (let [X (parse-to-x-matrix file-path)]
    (matrix/matrix (matrix/emap type-func X))))

(def int-parser #(Integer/parseInt %))

(def double-parser #(Double/parseDouble %))
