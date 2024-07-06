"use client";

import { useState } from "react";

const CategoryInput = ({ label, categoryList, setCategoryList, error }) => {
  const [inputValue, setInputValue] = useState("");

  const handleRemoveCategory = (categoryToRemove) => {
    setCategoryList(
        categoryList.filter((category) => category !== categoryToRemove)
    );
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
    e.preventDefault()
    const trimmedValue = inputValue.trim()
      if (trimmedValue && !categoryList.includes(trimmedValue)) {
        setCategoryList((prevList) => [...prevList, trimmedValue]);
        setInputValue("");
      }
    }
  };

  return (
    <div>
      <label htmlFor="category-box" className="block text-white my-1">
        {label}
      </label>
      <div className="flex flex-wrap gap-x-1">
        {categoryList.map((category, index) => (
          <span
            key={index}
            className="flex justify-center items-center bg-[#3FEABF] rounded-lg px-2 text-black"
          >
            {category}
            <button
              className="px-1"
              onClick={() => handleRemoveCategory(category)}
            >
              ×
            </button>
          </span>
        ))}
        <input
          type="text"
          id="category-box"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          className="text-white p-2 border border-white border-opacity-50 rounded-lg h-8 bg-transparent"
          placeholder="Add category"
        />
        {error && <p className="text-red-500 text-sm">{error}</p>}
      </div>
    </div>
  );
};

export default CategoryInput;
