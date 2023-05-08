import React from "react";
import { Button } from "@material-ui/core";
import Grid from "@material-ui/core/grid";
import Dialog from "@material-ui/core/Dialog";
import {
  createStyles,
  Theme,
  withStyles,
  WithStyles,
} from "@material-ui/core/styles";
import MuiDialogTitle from "@material-ui/core/DialogTitle";
import MuiDialogContent from "@material-ui/core/DialogContent";
import MuiDialogActions from "@material-ui/core/DialogActions";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import Typography from "@material-ui/core/Typography";

const styles = (theme: Theme) =>
  createStyles({
    root: {
      margin: 0,
      padding: theme.spacing(2),
    },
    closeButton: {
      position: "absolute",
      right: theme.spacing(1),
      top: theme.spacing(1),
      color: theme.palette.grey[500],
    },
  });

export interface DialogTitleProps extends WithStyles<typeof styles> {
  id: string;
  children: React.ReactNode;
  onClose: () => void;
}

const DialogTitle = withStyles(styles)((props: DialogTitleProps) => {
  const { children, classes, onClose, ...other } = props;
  return (
    <MuiDialogTitle disableTypography className={classes.root} {...other}>
      <Typography variant="h6">{children}</Typography>
      {onClose ? (
        <IconButton
          aria-label="close"
          className={classes.closeButton}
          onClick={onClose}
        >
          <CloseIcon />
        </IconButton>
      ) : null}
    </MuiDialogTitle>
  );
});

const DialogContent = withStyles((theme: Theme) => ({
  root: {
    padding: theme.spacing(2),
  },
}))(MuiDialogContent);

const DialogActions = withStyles((theme: Theme) => ({
  root: {
    margin: 0,
    padding: theme.spacing(1),
  },
}))(MuiDialogActions);

export default function ResultButtons(props) {
  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  function handleHide() {
    props.props.props.handleSubmit();
  }

  return (
    <div>
      <Grid container spacing={3} style={{marginBottom: '30px'}}>
        <Grid item xs={3} />
        <Grid item xs={3}>
          <Button variant="contained" size="large" onClick={handleHide}>
            ย้อนกลับ
          </Button>
        </Grid>
        <Grid item xs={3}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            onClick={handleClickOpen}
          >
            ติดต่อทนาย
          </Button>
          <Dialog
            onClose={handleClose}
            aria-labelledby="customized-dialog-title"
            open={open}
          >
            <DialogTitle id="customized-dialog-title" onClose={handleClose}>
              ประกาศ
            </DialogTitle>
            <DialogContent dividers>
              <Typography gutterBottom>
              ขณะนี้ทนายกำลังให้บริการเต็มหมดทุกคู่สาย กรุณารอซักครู่
              </Typography>
            </DialogContent>
            <DialogActions>
              <Button autoFocus onClick={handleClose} color="primary">
                ปิดหน้าต่าง
              </Button>
            </DialogActions>
          </Dialog>
        </Grid>
        <Grid item xs={3} />
      </Grid>
    </div>
  );
}
