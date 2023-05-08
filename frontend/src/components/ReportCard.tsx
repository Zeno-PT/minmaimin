import React from "react";
import Paper from "@material-ui/core/Paper";
import { Theme, createStyles, makeStyles } from "@material-ui/core/styles";
import { Report } from "../model/report.model";
import Typography from "@material-ui/core/Typography";

function dateFormat(date: Date): string {
  return date.toLocaleDateString("th-TH", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      margin: 0,
      padding: "0% 7.5% 5% 7.5%",
    },
    paper: {
      padding: "5px 20px",
      borderRadius: 10,
    },
  })
);
function ReportCard(props) {
  const classes = useStyles();
  return (
    <div className={classes.root}>
      <Paper elevation={3} className={classes.paper}>
        <p style={{ fontWeight: "bold" }}>รายงานผล</p>
        <p>
          กฎหมายที่เกี่ยวข้อง: {props.title ?? "ประมวลกฎหมายอาญา มาตรา 326"}
        </p>
        <p>
          ดำเนินการฟ้องร้องได้ถึงวันที่<span> </span>
          {/* {dateFormat(props.prescribedDate ?? new Date())} */}
          {dateFormat(props.props.props.newDate)}
        </p>
        <Typography
          color="textSecondary"
          gutterBottom
          style={{ fontSize: "small" }}
        >
          หมายเหตุ: กฎหมายที่ควรพิจารณา{" "}
          {props.otherLawSections ??
            "ประมวลกฎหมายอาญา มาตรา 328 (หมื่นประมาทโดยการโฆษณา), พ.ร.บ.คอมพิวเตอร์ฯ มาตรา 14"}
        </Typography>
      </Paper>
    </div>
  );
}

export default ReportCard;
