from django.db import models
from django.contrib.auth.base_user import BaseUserManager, AbstractBaseUser
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
import uuid

class MyUserManager(BaseUserManager):
    def create_user(self, email, password=None, **kwargs):
        """
        Creates and saves a User with the given email, date of
        birth and password.
        """
        if not email:
            raise ValueError('Users must have an email address')
        if password is None:
            password = 'f1a28a8a-8308-4c03-aef8-d07a61136807' # Secret Key

        user = self.model(
            email=self.normalize_email(email),
			first_name=kwargs.get('first_name', ""),
            last_name=kwargs.get('last_name', "")
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password, **kwargs):
        """
        Creates and saves a superuser with the given email, date of
        birth and password.
        """
        user = self.create_user(email,
            password=password,
			first_name=kwargs.get('first_name', ""),
            last_name=kwargs.get('last_name', "")
        )
        user.is_admin = True
        user.save(using=self._db)
        return user


class MyUser(AbstractBaseUser):
    email = models.EmailField(
        verbose_name='email address',
        max_length=255,
        unique=True,
    )

    first_name = models.CharField(max_length=40, blank=True)
    last_name = models.CharField(max_length=40, blank=True)

    default_tzid = models.CharField(max_length=50) # ex. Europe/London - http://www.iana.org/time-zones

    date_joined = models.DateTimeField(default=timezone.now)

    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)

    objects = MyUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def get_full_name(self):
        # The user is identified by their email address
        return self.email

    def get_short_name(self):
        # The user is identified by their email address
        return self.email

    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        # Simplest possible answer: Yes, always
        return True

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        # Simplest possible answer: Yes, always
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        # Simplest possible answer: All admins are staff
        return self.is_admin


class Profile(models.Model):

    """
    Additional data about a user
    """

    user = models.OneToOneField(MyUser)
    profile_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    contact_number = models.CharField(max_length=20)
    status_message = models.CharField(
        max_length=144, help_text='Twitter style status message', blank=True, null=True)
    bio = models.TextField(blank=True, null=True)


# class AppAuthorization(models.Model): # WebHooks
#
#     """
#     Collection of authentication tokens/keys for various services
#     """
#
#     user = models.ForeignKey(MyUser, related_name='authentications')
#     service_name = models.CharField(
#         max_length=20, help_text='Service such as slack, hipchat, etc.')
#     key = models.CharField(max_length=200, blank=True, null=True,
#                            help_text='Optional, typically your API_KEY value')
#     token = models.CharField(
#         max_length=200, blank=True, null=True, help_text='Optional, this is your token or secret')


@receiver(post_save, sender=MyUser)
def new_user_created(sender, instance, **kwargs):

    if kwargs.get("created", False):
        Profile.objects.create(user=instance)


def generate_client_id():
    return uuid.uuid4()


def generate_client_secret():
    return uuid.uuid4()


class Application(models.Model):
    """
    An Application instance represents a Client on the Authorization server.
    Usually an Application is created manually by client's developers after
    logging in on an Authorization Server.
    Fields:
    * :attr:`client_id` The client identifier issued to the client during the
                        registration process as described in :rfc:`2.2`
    * :attr:`user` ref to a Django user
    * :attr:`redirect_uris` The list of allowed redirect uri. The string
                            consists of valid URLs separated by space
    * :attr:`client_type` Client type as described in :rfc:`2.1`
    * :attr:`authorization_grant_type` Authorization flows available to the
                                       Application
    * :attr:`client_secret` Confidential secret issued to the client during
                            the registration process as described in :rfc:`2.2`
    * :attr:`name` Friendly name for the Application
    """
    owner = models.ForeignKey(MyUser)
    name = models.CharField(max_length=100)
    url = models.CharField(max_length=70)
    logo = models.FileField(null=True, blank=True)
    client_id = models.CharField(max_length=100, unique=True,
                                 default=generate_client_id, db_index=True)
    client_secret = models.CharField(max_length=255, blank=True,
                                     default=generate_client_secret, db_index=True)

    def __str__(self):
        return self.name or self.client_id


class ApplicationUser(models.Model):
    application = models.ForeignKey(Application)
    user = models.ForeignKey(MyUser)
    scope = models.TextField(null=True) # read_account list_calendars read_events create_event delete_event read_free_busy
