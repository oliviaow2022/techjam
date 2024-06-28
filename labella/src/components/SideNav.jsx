export default function SideNav({params}){
    const menuOptions = [{
        "id": 0,
        "name": "Create Project",
        "link": "/image-classification"
    },
    {
        "id": 3,
        "name": "Label",
        "link": `/label/${params}`
    },
    {
        "id": 4,
        "name": "Train Model",
        "link": `/train-model/${params}`
    },
    {
        "id": 5,
        "name": "Performance and Statistics",
        "link": `/statistics/${params}`
    }]
    return(<div className="hidden lg:flex lg:flex-col gap-4 mr-8 pt-2 fixed top-40 z-10">
        {menuOptions.map((option, index) => (
            <p key={index} className="hover:cursor-pointer hover:text-[#FF52BF] text-white"><a href={option.link}>{option.name}</a></p>
        ))}
    </div>)
}