const OptionsInput = ({ label, items, noItemsMessage, bgColour, selectedItem, setSelectedItem, renderItem, error }) => {
  return (
    <div>
      <p className="font-bold mb-2">{label}</p>
      {items.length === 0 && <p>{noItemsMessage}</p>}
      <div className="mb-4 grid lg:grid-cols-2 gap-x-6 gap-y-1">
        {items.map((item, index) => (
          <div
            key={index}
            className={`flex flex-wrap border border-white border-opacity-50 w-72 items-center justify-center rounded-lg h-8 cursor-pointer my-1 ${
              selectedItem === item
                ? `${bgColour} text-black`
                : `hover:${bgColour} hover:text-black`
            }`}
            onClick={setSelectedItem(item)}
          >
            {renderItem ? renderItem(item) : item}
          </div>
        ))}
      </div>
      {error && <p className="text-red-500 text-sm">{error}</p>}
    </div>
  );
};

export default OptionsInput;
