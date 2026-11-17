"""Study blocks for the multi-area test.

Each block is 10.5 x 8.4 km in EPSG:3157 (NAD83(CSRS) / UTM 10N), bounds in
multiples of 30 m, and fully covered by the public NRCan lidar mosaic. They were
picked by a coverage search, not by terrain knowledge, so they are named by
position. `range` records which mountain system the block is in.
"""

SITES = {
    # id: (west, south, east, north), label, range
    "mamquam": ((510600.0, 5510700.0, 521100.0, 5519100.0),
                "49.78N 122.78W, east of the Mamquam Icefield", "Coast Mountains"),
    "coast_4986": ((449940.0, 5518890.0, 460440.0, 5527290.0),
                   "49.86N 123.62W, Sunshine Coast backcountry", "Coast Mountains"),
    "coast_5036": ((421920.0, 5575410.0, 432420.0, 5583810.0),
                   "50.36N 124.02W, inlet country north of Powell River", "Coast Mountains"),
    "coast_5066": ((418860.0, 5608020.0, 429360.0, 5616420.0),
                   "50.66N 124.07W, inlet country north of Powell River", "Coast Mountains"),
    "marble_5122": ((570150.0, 5670570.0, 580650.0, 5678970.0),
                    "51.22N 121.92W, Marble Range", "Interior, east of the Fraser River"),
}
