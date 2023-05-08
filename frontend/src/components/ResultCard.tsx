import React from "react";
import GaugeMeter from "./GaugeMeter";
import ResultButtons from "./ResultButtons";
import ReportCard from "./ReportCard";
import { Theme, createStyles, makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      width: "75%",
      alignItems: "center",
      margin: "auto auto",
    },
  })
);

export default function ResultCard(props) {
  const classes = useStyles();
  return (
    <div className={classes.root}>
      <GaugeMeter />
      <ReportCard props={props}/>
      <ResultButtons props={props}/>
    </div>
  );
}
