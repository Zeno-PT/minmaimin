import React from "react";
import Gauge from "react-svg-gauge";

interface AppProps {}

interface AppState {
  value: number;
}

export default class GaugeMeter extends React.Component<AppProps, AppState> {
  constructor(props: AppProps) {
    super(props);
    this.state = {
      value: 84,
    };
  }
  render() {
    const revertedValue = 100 - this.state.value;
    let r = Math.floor(revertedValue * 2.55);
    let g = Math.floor(255 - revertedValue * 2.55);
    let b = 0;
    let colorHex = "#" + this.getHexColor(r) + this.getHexColor(g) + this.getHexColor(b);

    return (
      <div>
        <Gauge
          value={this.state.value}
          width={480}
          height={320}
          color={colorHex}
          topLabelStyle={{color: 'black', fontSize: '28px', fontWeight: 'bold'}}
          valueLabelStyle={{fontSize: '73px', fontWeight: 'bold'}}
          valueFormatter={this.formatter}
          label="โอกาสเข้าข่ายมีความผิดหมิ่นประมาท"
        />
      </div>
    );
  }

  private formatter(val: number): string {
    return `${val.toString()}%`;
  }

  private getHexColor(value: number): string {
    const string = value.toString(16);
    return string.length === 1 ? "0" + string : string;
  }
}
