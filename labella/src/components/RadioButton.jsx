'use client'
import React, { useState } from 'react';

export default function RadioButton({ optionName, selectedOption, handleOptionChange }) {
    return (
        <div>
            <label htmlFor="input-box" className="mb-2">
                <input
                    type="radio"
                    name={optionName}
                    value={optionName}
                    checked={selectedOption === optionName}
                    onChange={handleOptionChange}
                    className="mr-2"
                />{optionName}</label>

        </div>
    )
}