export default function Arrow({ direction }) {
  return (
    <div>
      {direction === "left" ? (
        <img
          src="/left-arrow.png"
          alt="left-arrow"
          className="block object-contain w-10"
        />
      ) : (
        <img
          src="/right-arrow.png"
          alt="right-arrow"
          className="block object-contain w-10"
        />
      )}
    </div>
  );
}
