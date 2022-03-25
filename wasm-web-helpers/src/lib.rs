/// Parse the query string as returned by `web_sys::window()?.location().search()?` and get a
/// specific key out of it.
pub fn parse_url_query_string<'a>(query: &'a str, search_key: &str) -> Option<&'a str> {
    let query_string = query.strip_prefix('?')?;

    for pair in query_string.split('&') {
        let mut pair = pair.split('=');
        let key = pair.next()?;
        let value = pair.next()?;

        if key == search_key {
            return Some(value);
        }
    }

    None
}

pub fn parse_url_query_string_from_window(search_key: &str) -> Option<String> {
    let query_string = web_sys::window().unwrap().location().search().unwrap();
    parse_url_query_string(&query_string, search_key).map(|string| string.to_owned())
}

pub fn append_canvas(window: &winit::window::Window) {
    use winit::platform::web::WindowExtWebSys;

    web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
}