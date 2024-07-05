export default function ClassificationImageClassificationSideNav({ params }) {
  const menuOptions = [
    {
      id: 0,
      name: "Label Data",
      link: `/image-classification/${params}/label`,
    },
    {
      id: 1,
      name: "Train Model",
      link: `/image-classification/${params}/train-model`,
    },
    {
      id: 2,
      name: "Model Performance",
      link: `/image-classification/${params}/statistics`,
    },
    {
      id: 3,
      name: "Run Model",
      link: `/image-classification/${params}/run-model`,
    },
  ];
  return (
    <>
      <div className="hidden lg:flex lg:flex-col gap-4 mr-8 pt-2 fixed top-40 z-10">
        {menuOptions.map((option, index) => (
          <p
            key={index}
            className="hover:cursor-pointer hover:text-[#FF52BF] text-white"
          >
            <a href={option.link}>{option.name}</a>
          </p>
        ))}
      </div>
      <div className="bg-white border-2 mt-32 ml-64 h-100% hidden lg:block"></div>
    </>
  );
}
