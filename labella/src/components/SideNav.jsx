export default function SideNav({ params }) {
  const menuOptions = [
    {
      id: 0,
      name: "Label Data",
      link: `/label/${params}`,
    },
    {
      id: 1,
      name: "Train Model",
      link: `/train-model/${params}`,
    },
    {
      id: 2,
      name: "Model Performance",
      link: `/statistics/${params}`,
    },
    {
      id: 3,
      name: "Run Model",
      link: `/run-model/${params}`,
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
