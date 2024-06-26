export default function SideNav({params}){
    const menuOptions = [{
        "id": 0,
        "name": "Project Details",
        "link": "/image-classification"
    },
    {
        "id": 1,
        "name": "Model",
        "link": "/image-classification"
    },
    {
        "id": 2,
        "name": "Dataset",
        "link": "/image-classification"
    },
    {
        "id": 3,
        "name": "Label",
        "link": `/label/${params}`
    },
    {
        "id": 5,
        "name": "Performance and Statistics",
        "link": `/statistics/${params}`
    }]
    return(<div className="hidden lg:flex lg:flex-col gap-4 mr-8 pt-2 fixed top-40 z-10">
        {menuOptions.map((option, index) => (
            <p key={index} className="hover:cursor-pointer hover:text-[#FF52BF] text-white"><a href={`${option.link}#${option.name}`}>{option.name}</a></p>
        ))}
    </div>)
}