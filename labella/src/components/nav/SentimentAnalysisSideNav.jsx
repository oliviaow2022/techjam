import Link from 'next/link';
import { usePathname } from 'next/navigation'

export default function SentimentAnalysisSideNav({ params }) {
  const currentPath = usePathname();
  const menuOptions = [
    {
      id: 0,
      name: "Label Data",
      link: `/sentiment-analysis/${params}/label`,
    },
    {
      id: 1,
      name: "Train Model",
      link: `/sentiment-analysis/${params}/train-model`,
    },
    {
      id: 2,
      name: "Model Performance",
      link: `/sentiment-analysis/${params}/statistics`,
    }
  ];
  return (
    <>
      <div className="hidden lg:flex lg:flex-col gap-4 mr-8 pt-2 fixed top-40 z-10">
        {menuOptions.map((option, index) => (
          <p
            key={index}
            className={`hover:cursor-pointer ${currentPath === option.link ? 'text-[#3FEABF]' : 'text-white'}`}
          >
            <Link href={option.link}>{option.name}</Link>
          </p>
        ))}
      </div>
      <div className="bg-white border-2 mt-32 ml-64 h-100% hidden lg:block"></div>
    </>
  );
}
