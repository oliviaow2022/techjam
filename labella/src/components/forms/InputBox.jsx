'use client';
import React, {useState} from 'react';

const InputBox = ({label, name, value, onChange, error}) => {

    return(
        <div>
            <label htmlFor={`input-box-${name}`} className='block text-white my-1'>{label}</label>
            <input 
                id={`input-box-${name}`}
                name={name}
                type="type" 
                value={value}
                onChange={onChange} 
                className='text-white p-2 border border-white border-opacity-50 rounded-lg h-8 bg-transparent'    
            />
            {error && <p className="text-red-500 text-sm">{error}</p>}
        </div>
    )
};

export default InputBox;