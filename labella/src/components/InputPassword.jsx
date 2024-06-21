import React, { useState } from 'react';

const InputPassword = ({label, name, value, onChange, error}) => {
    const [showPassword, setShowPassword] = useState(false);

    const togglePasswordVisibility = () => {
        setShowPassword(!showPassword);
    };

    return (
        <div>
            <label htmlFor={name} className='block'>{label}</label>
            <div className='relative'>
                <input
                    id={name}
                    name={name}
                    type={showPassword ? 'text' : 'password'} // Toggle between 'text' and 'password'
                    value={value}
                    onChange={onChange}
                    className='text-white p-2 border border-white border-opacity-50 rounded-lg h-8 bg-transparent'
                />
                <button
                    type='button'
                    className='absolute inset-y-0 right-0 px-2 py-1 text-sm text-[#ACDFEF]'
                    onClick={togglePasswordVisibility}
                >
                    {showPassword ? 'Hide' : 'Show'}
                </button>
            </div>
            {error && <p className="text-red-500 text-sm">{error}</p>}
        </div>
    );
};

export default InputPassword;
