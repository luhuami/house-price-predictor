(ns linear-regression.linear-regression
  (:require [clojure.core.matrix :as matrix]))

(defn- same-dimension? [X y]
  (let [x-shape (matrix/shape X)
        y-shape (matrix/shape y)]
    (= (first x-shape) (first y-shape))))

(defn- create-bias-column [m]
  (matrix/broadcast [1] [m 1]))

(defn- add-bias [X]
  (if (matrix/vec? X)
    (matrix/join [1] X)
    (let [bias-column (create-bias-column (first (matrix/shape X)))]
      (matrix/join-along 1 bias-column X))))