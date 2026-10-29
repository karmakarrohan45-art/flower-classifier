import React from "react";
import "./LoadingSpinner.css";

const LoadingSpinner: React.FC<{ message: string }> = ({ message }) => (
  <div className="loading-spinner">
    <div className="spinner" />
    <p>{message}</p>
  </div>
);

export default LoadingSpinner;
